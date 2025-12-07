# Delta-Net Kernel Implementation for AMD MI50 (gfx906)

## Executive Summary

This document describes a complete GPU kernel implementation for **Delta-Net linear attention**, optimized for AMD MI50 (gfx906) GPUs. Delta-Net is a linear attention mechanism used in hybrid models like **Qwen3-Next**, which combines Mamba-2 style state-space models with traditional transformer attention.

The implementation uses an **incremental validation strategy** across 8 phases, with 26 total tests validating all kernel operations against host reference implementations.

---

## Table of Contents

1. [Background: What is Delta-Net?](#1-background-what-is-delta-net)
2. [Architecture Overview](#2-architecture-overview)
3. [Target Hardware Constraints](#3-target-hardware-constraints)
4. [Implementation Phases](#4-implementation-phases)
5. [Kernel Reference](#5-kernel-reference)
6. [Algorithm Deep Dive](#6-algorithm-deep-dive)
7. [Test Framework](#7-test-framework)
8. [Building and Running](#8-building-and-running)
9. [Performance Considerations](#9-performance-considerations)
10. [Future Optimization Opportunities](#10-future-optimization-opportunities)

---

## 1. Background: What is Delta-Net?

### 1.1 Linear Attention vs Traditional Attention

Traditional transformer attention has **O(n²)** complexity with sequence length, making it expensive for long sequences. Linear attention mechanisms like Delta-Net achieve **O(n)** complexity by reformulating attention as a recurrent state update.

### 1.2 Delta-Net Core Concept

Delta-Net uses a **delta rule** for memory updates, similar to Hebbian learning:

```
state_new = state * decay + key ⊗ value
```

Where:
- `state` is a [S, S] matrix storing key-value associations
- `decay` is an exponential decay factor from gating
- `key ⊗ value` is the outer product update (rank-1 update)

### 1.3 WY Representation

The key innovation in Delta-Net is using **WY representation** (from Householder QR factorization) to parallelize within chunks. Instead of sequential state updates, we can express the intra-chunk computation as:

```
(I - A) × X = B
```

Where `A` is a strictly lower triangular attention matrix. This triangular system can be solved via **forward substitution**, enabling parallel computation within each chunk.

### 1.4 Chunked Parallelism

Delta-Net processes sequences in chunks of 64 tokens:
- **Intra-chunk**: Parallelized via WY representation (triangular solve)
- **Inter-chunk**: Sequential state updates between chunks

This achieves a good balance between parallelism and memory efficiency.

---

## 2. Architecture Overview

### 2.1 Dimensions (Qwen3-Next)

```cpp
constexpr int S_K = 128;          // Key head dimension (ssm_d_state)
constexpr int S_V = 128;          // Value head dimension
constexpr int CHUNK_SIZE = 64;    // Tokens per chunk
constexpr int NUM_K_HEADS = 16;   // Number of K heads (ssm_n_group)
constexpr int NUM_V_HEADS = 32;   // Number of V heads (ssm_dt_rank)
```

### 2.2 Input Tensors

| Tensor | Shape | Description |
|--------|-------|-------------|
| Q | [S, n_tokens] | Queries |
| K | [S, n_tokens] | Keys |
| V | [S, n_tokens] | Values |
| G | [n_tokens] | Gate values (for decay) |
| Beta | [n_tokens] | Write strength (sigmoided) |
| State | [S, S] | Recurrent state matrix |

### 2.3 Output Tensors

| Tensor | Shape | Description |
|--------|-------|-------------|
| Output | [S, n_tokens] | Attention output |
| State_out | [S, S] | Updated state |

---

## 3. Target Hardware Constraints

### 3.1 AMD MI50 (gfx906) Specifications

| Resource | Limit |
|----------|-------|
| LDS (Shared Memory) | 64 KB per workgroup |
| Wavefront size | 64 threads |
| Max workgroup size | 1024 threads |
| On-device printf | **Not supported** |
| Tensor cores | **Not available** |

### 3.2 Key Constraints

1. **No on-device printf**: All debugging must be done via host-side comparison. This necessitated our validation-driven development approach.

2. **64 KB LDS limit**: Constrains tile sizes for GEMM operations. We use 16×16 tiles for tiled GEMM.

3. **64-wide wavefronts**: Operations should be designed with 64-thread granularity for efficiency.

4. **FP32 only**: No hardware FP16 tensor core acceleration, so all operations use FP32.

---

## 4. Implementation Phases

### Phase 1: Element-wise Operations

**Purpose**: Basic building blocks for Delta-Net computation.

| Kernel | Formula | Tests |
|--------|---------|-------|
| `kernel_sigmoid` | out[i] = 1/(1 + exp(-in[i])) | 1024 elements |
| `kernel_exp` | out[i] = exp(in[i]) | 1024 elements |
| `kernel_mul` | out[i] = a[i] × b[i] | 1024 elements |
| `kernel_cumsum_sequential` | out[i] = Σ(in[0..i]) | 64 elements |

**Implementation Notes**:
- All use 1D thread indexing: `idx = blockIdx.x * blockDim.x + threadIdx.x`
- Cumsum uses sequential single-thread implementation for validation (parallel scan would be used in production)

```cpp
__global__ void kernel_sigmoid(const float* in, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = 1.0f / (1.0f + expf(-in[idx]));
    }
}
```

---

### Phase 2: Triangular Operations

**Purpose**: Matrix operations required for causal masking and triangular solve.

| Kernel | Description | Output |
|--------|-------------|--------|
| `kernel_tril` | Extract lower triangular | Zeros above diagonal |
| `kernel_causal_mask` | Create causal mask | 1s where col < row |
| `kernel_eye` | Identity matrix | 1s on diagonal |
| `kernel_solve_tri_sequential` | Forward substitution | Solve (I-A)X = B |

**Triangular Solve Algorithm**:

The triangular solve is the **core of Delta-Net's parallelization**. Given:
- `A`: n×n strictly lower triangular matrix
- `B`: n×k right-hand side
- Solve: `(I - A) × X = B`

Forward substitution:
```cpp
for (int col = 0; col < k; col++) {
    for (int row = 0; row < n; row++) {
        float sum = B[row * k + col];
        for (int j = 0; j < row; j++) {
            sum += A[row * n + j] * X[j * k + col];
        }
        X[row * k + col] = sum;
    }
}
```

**Key Insight**: Since `(I - A)` has 1s on the diagonal, no division is needed.

---

### Phase 3: Matrix Operations (GEMM)

**Purpose**: Matrix multiplications for attention computation.

| Kernel | Operation | Use Case |
|--------|-----------|----------|
| `kernel_gemm_nn_naive` | C = A @ B | V_new computation |
| `kernel_gemm_nt_naive` | C = A @ B^T | K_cumdecay, state update |
| `kernel_gemm_tn_naive` | C = A^T @ B | Attention matrix |
| `kernel_gemm_nt_tiled` | C = A @ B^T (LDS tiled) | Optimized state update |

**Naive GEMM Implementation**:
```cpp
__global__ void kernel_gemm_nt_naive(
    const float* A, const float* B, float* C,
    int M, int N, int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[col * K + k];  // B transposed
        }
        C[row * N + col] = sum;
    }
}
```

**Tiled GEMM with LDS**:
```cpp
#define TILE_SIZE 16

__global__ void kernel_gemm_nt_tiled(
    const float* A, const float* B, float* C,
    int M, int N, int K
) {
    __shared__ float s_A[TILE_SIZE][TILE_SIZE];
    __shared__ float s_B[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load tiles to LDS
        int a_col = t * TILE_SIZE + threadIdx.x;
        s_A[threadIdx.y][threadIdx.x] = (row < M && a_col < K)
            ? A[row * K + a_col] : 0.0f;

        int b_col = t * TILE_SIZE + threadIdx.y;
        s_B[threadIdx.y][threadIdx.x] = (col < N && b_col < K)
            ? B[col * K + b_col] : 0.0f;

        __syncthreads();

        // Compute partial dot product
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += s_A[threadIdx.y][k] * s_B[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}
```

---

### Phase 4: Decay Mask Computation

**Purpose**: Compute exponential decay mask for causal attention.

**Formula**:
```
decay_mask[i][j] = exp(g_cumsum[j] - g_cumsum[i])  if j ≤ i
                 = 0                                 otherwise
```

**Kernel**:
```cpp
__global__ void kernel_decay_mask(
    const float* g_cumsum, float* mask, int n
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        if (col <= row) {
            mask[row * n + col] = expf(g_cumsum[col] - g_cumsum[row]);
        } else {
            mask[row * n + col] = 0.0f;
        }
    }
}
```

**Physical Interpretation**: The decay mask creates an exponential falloff where earlier positions (smaller `j`) have decayed more relative to later positions (larger `i`).

---

### Phase 5: Attention Matrix Construction

**Purpose**: Build the attention matrix before triangular solve.

**Formula**:
```
attn = -(K^T @ K_beta) × decay_mask × causal_mask
```

**Fused Kernel**:
```cpp
__global__ void kernel_attention_matrix_fused(
    const float* K, const float* K_beta,
    const float* decay_mask, float* attn,
    int S, int C
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < C && col < C) {
        // Compute K^T @ K_beta at [row, col]
        float sum = 0.0f;
        for (int s = 0; s < S; s++) {
            sum += K[s * C + row] * K_beta[s * C + col];
        }

        // Apply decay mask and causal mask (strictly lower tri)
        if (col < row) {
            attn[row * C + col] = -sum * decay_mask[row * C + col];
        } else {
            attn[row * C + col] = 0.0f;
        }
    }
}
```

---

### Phase 6: Full Intra-Chunk Computation

**Purpose**: Complete computation within a single chunk.

**Steps**:
1. `beta_sig = sigmoid(Beta)`
2. `g_cumsum = cumsum(G)`
3. `decay_mask = compute_decay_mask(g_cumsum)`
4. `K_beta = K × beta_sig` (broadcast)
5. `V_beta = V × beta_sig` (broadcast)
6. `attn_pre = -(K^T @ K_beta) × decay_mask × causal_mask`
7. `attn_solved = solve_tri(attn_pre, attn_pre)` (forward substitution)
8. `attn_solved = apply_causal_identity(attn_solved)` (add I on diagonal)
9. `V_new = V_beta @ attn_solved^T`
10. `K_cumdecay = (attn_solved @ (K_beta × exp(g_cumsum))^T)^T`

**GPU Kernel Sequence**:
```cpp
// Step 1-2: Sigmoid and cumsum
kernel_sigmoid<<<...>>>(d_Beta, d_beta_sig, C);
kernel_cumsum_sequential<<<1, 1>>>(d_G, d_g_cumsum, C);

// Step 3: Decay mask
kernel_decay_mask<<<grid_C, block2d>>>(d_g_cumsum, d_decay_mask, C);

// Step 4-5: Broadcast multiply
kernel_broadcast_mul<<<...>>>(d_K, d_beta_sig, d_K_beta, S, C);
kernel_broadcast_mul<<<...>>>(d_V, d_beta_sig, d_V_beta, S, C);

// Step 6: Attention matrix
kernel_gemm_tn_naive<<<...>>>(d_K, d_K_beta, d_kmulkbeta, C, C, S);
kernel_mul<<<...>>>(d_kmulkbeta, d_decay_mask, d_k_decay, C * C);
kernel_causal_mask<<<...>>>(d_causal, C);
kernel_mul<<<...>>>(d_k_decay, d_causal, d_attn_pre, C * C);
kernel_neg<<<...>>>(d_attn_pre, d_attn_pre, C * C);

// Step 7-8: Triangular solve + identity
kernel_solve_tri_sequential<<<1, 1>>>(d_attn_pre, d_attn_pre, d_attn_solved, C, C);
kernel_apply_causal_identity<<<...>>>(d_attn_solved, d_attn_solved, C);

// Step 9: V_new = V_beta @ attn_solved^T
kernel_transpose<<<...>>>(d_attn_solved, d_attn_T, C, C);
kernel_gemm_nn_naive<<<...>>>(d_V_beta, d_attn_T, d_V_new, S, C, C);

// Step 10: K_cumdecay
kernel_exp<<<...>>>(d_g_cumsum, d_gexp, C);
kernel_broadcast_mul<<<...>>>(d_K_beta, d_gexp, d_kbeta_gexp, S, C);
kernel_gemm_nt_naive<<<...>>>(d_attn_solved, d_kbeta_gexp, d_temp, C, S, C);
kernel_transpose<<<...>>>(d_temp, d_K_cumdecay, C, S);
```

---

### Phase 7: State Interaction

**Purpose**: Compute output using state and update state.

**Output Computation**:
```
output = Q_gexp @ State^T + attn @ V_new
```

**State Update**:
```
state_new = state × exp(g_last) + K_cumdecay @ V_new^T
```

**Kernels**:
```cpp
// State update
kernel_scale<<<...>>>(d_state, d_state_scaled, S * S, exp_g_last);
kernel_gemm_nt_tiled<<<...>>>(d_K_cumdecay, d_V_new, d_kv, S, S, C);
kernel_add<<<...>>>(d_state_scaled, d_kv, d_state, S * S);
```

---

### Phase 8: Full Multi-Chunk Delta-Net

**Purpose**: Process multiple chunks with state propagation.

**Algorithm**:
```cpp
for (int chunk = 0; chunk < n_chunks; chunk++) {
    // Extract chunk data
    K_chunk = K[:, chunk*C : (chunk+1)*C]
    V_chunk = V[:, chunk*C : (chunk+1)*C]
    G_chunk = G[chunk*C : (chunk+1)*C]
    Beta_chunk = Beta[chunk*C : (chunk+1)*C]

    // Intra-chunk computation
    intra_chunk(K_chunk, V_chunk, G_chunk, Beta_chunk,
                -> attn_solved, V_new, K_cumdecay)

    // State update
    g_last = g_cumsum[C-1]
    state = state * exp(g_last) + K_cumdecay @ V_new^T
}
```

---

## 5. Kernel Reference

### 5.1 Element-wise Kernels

| Kernel | Signature | Description |
|--------|-----------|-------------|
| `kernel_sigmoid` | `(in, out, n)` | σ(x) = 1/(1+e^(-x)) |
| `kernel_exp` | `(in, out, n)` | e^x |
| `kernel_mul` | `(a, b, out, n)` | a × b |
| `kernel_add` | `(a, b, out, n)` | a + b |
| `kernel_sub` | `(a, b, out, n)` | a - b |
| `kernel_neg` | `(in, out, n)` | -x |
| `kernel_scale` | `(in, out, n, s)` | x × s |
| `kernel_cumsum_sequential` | `(in, out, n)` | Prefix sum |

### 5.2 Matrix Kernels

| Kernel | Signature | Description |
|--------|-----------|-------------|
| `kernel_broadcast_mul` | `(a, b, out, S, C)` | out[s,c] = a[s,c] × b[c] |
| `kernel_transpose` | `(in, out, rows, cols)` | out[j,i] = in[i,j] |
| `kernel_gemm_nn_naive` | `(A, B, C, M, N, K)` | C = A @ B |
| `kernel_gemm_nt_naive` | `(A, B, C, M, N, K)` | C = A @ B^T |
| `kernel_gemm_tn_naive` | `(A, B, C, M, N, K)` | C = A^T @ B |
| `kernel_gemm_nt_tiled` | `(A, B, C, M, N, K)` | C = A @ B^T (LDS) |

### 5.3 Triangular Kernels

| Kernel | Signature | Description |
|--------|-----------|-------------|
| `kernel_tril` | `(src, dst, n, diag)` | Lower triangular |
| `kernel_causal_mask` | `(out, n)` | Strictly lower tri (1s) |
| `kernel_eye` | `(out, n)` | Identity matrix |
| `kernel_solve_tri_sequential` | `(A, B, X, n, k)` | Solve (I-A)X = B |
| `kernel_apply_causal_identity` | `(in, out, n)` | Add I, zero upper |

### 5.4 Attention Kernels

| Kernel | Signature | Description |
|--------|-----------|-------------|
| `kernel_decay_mask` | `(g_cumsum, mask, n)` | Exponential decay |
| `kernel_attention_matrix_fused` | `(K, K_beta, decay, attn, S, C)` | Build attn matrix |

---

## 6. Algorithm Deep Dive

### 6.1 The WY Representation

Delta-Net's parallelization comes from expressing the intra-chunk update as a matrix equation. Consider tokens 0, 1, ..., C-1 within a chunk.

The sequential update would be:
```
h[0] = state @ q[0] + v[0]
h[1] = state @ q[1] + attn[1,0] × h[0] + v[1]
h[2] = state @ q[2] + attn[2,0] × h[0] + attn[2,1] × h[1] + v[2]
...
```

This can be written as:
```
(I - A) × H = V + state @ Q^T
```

Where `A` is strictly lower triangular with `A[i,j] = -attn[i,j]` for j < i.

### 6.2 Forward Substitution

The system `(I - A) × X = B` with unit lower triangular `(I - A)` is solved by:

```
X[0] = B[0]
X[1] = B[1] + A[1,0] × X[0]
X[2] = B[2] + A[2,0] × X[0] + A[2,1] × X[1]
...
```

**Complexity**: O(n²) for single RHS, O(n² × k) for k columns.

### 6.3 Decay Mask Derivation

The decay comes from gating values `G[t]`:

```
g_cumsum[t] = Σ G[0..t]
decay[i,j] = exp(g_cumsum[j] - g_cumsum[i])
           = exp(g_cumsum[j]) / exp(g_cumsum[i])
           = Π exp(G[k]) for k in [j+1, i]
```

This represents the accumulated decay from position j to position i.

### 6.4 State Update Derivation

After processing a chunk, the state is updated:

```
state_new = state × exp(g_cumsum[C-1]) + Σ k[t] ⊗ v[t] × decay_to_end[t]
```

Where `decay_to_end[t] = exp(g_cumsum[C-1] - g_cumsum[t])`.

This is computed efficiently as:
```
state_new = state × exp(g_last) + K_cumdecay @ V_new^T
```

---

## 7. Test Framework

### 7.1 Test Structure

Each test follows this pattern:

```cpp
bool test_operation() {
    // 1. Allocate host arrays
    std::vector<float> h_in(...), h_ref(...), h_out(...);

    // 2. Initialize with random data
    fill_random(h_in.data(), N, min, max);

    // 3. Compute host reference
    host_operation(h_in.data(), h_ref.data(), N);

    // 4. Allocate device memory
    float* d_in = device_alloc<float>(N);
    float* d_out = device_alloc<float>(N);

    // 5. Copy to device
    to_device(d_in, h_in.data(), N);

    // 6. Launch kernel
    kernel_operation<<<grid, block>>>(d_in, d_out, N);
    HIP_CHECK(hipDeviceSynchronize());

    // 7. Copy back
    to_host(h_out.data(), d_out, N);

    // 8. Compare
    float diff = max_diff(h_ref.data(), h_out.data(), N);
    bool passed = diff < tolerance;

    // 9. Report
    report_test("test_name", passed, diff);
    return passed;
}
```

### 7.2 Tolerance Guidelines

| Operation | Tolerance | Reason |
|-----------|-----------|--------|
| Integer-like (eye, causal_mask) | 0 | Exact values |
| Simple FP (mul, add) | 1e-6 | Direct computation |
| Transcendental (sigmoid, exp) | 1e-5 | Library differences |
| Accumulated (GEMM, cumsum) | 1e-4 to 1e-3 | Floating point accumulation |
| Full pipeline | 1e-2 | Error accumulation |

### 7.3 Test Registry

```cpp
std::map<int, std::vector<std::string>> g_phases = {
    {1, {"sigmoid", "exp", "mul", "cumsum"}},
    {2, {"tril", "causal_mask", "eye", "solve_tri_small", "solve_tri_64"}},
    {3, {"gemm_nt_naive_small", "gemm_nt_naive_64", "gemm_tn_naive",
         "gemm_nt_tiled", "gemm_nt_tiled_128"}},
    {4, {"decay_mask_small", "decay_mask_64"}},
    {5, {"attn_matrix_small", "attn_matrix_64"}},
    {6, {"attn_pre_computation", "solve_causal_identity",
         "intra_chunk_small", "intra_chunk"}},
    {7, {"state_update", "output_with_state"}},
    {8, {"deltanet_single_chunk", "deltanet_full"}},
};
```

---

## 8. Building and Running

### 8.1 Prerequisites

- ROCm 5.x or later
- AMD MI50 GPU (gfx906)
- hipcc compiler

### 8.2 Build Commands

```bash
cd experiments/deltanet

# Build all
make

# Clean and rebuild
make clean all

# Debug build (no optimization)
make debug

# Check environment
make check-env
```

### 8.3 Running Tests

```bash
# Run all tests
./test_deltanet

# Run specific phase
./test_deltanet --phase 1
./test_deltanet --phase 8

# Run specific test
./test_deltanet --test sigmoid
./test_deltanet --test deltanet_full

# List available tests
./test_deltanet --list
```

### 8.4 Environment Variables

```bash
# Required for gfx906
export HSA_OVERRIDE_GFX_VERSION=9.0.6
export HIP_VISIBLE_DEVICES=0
```

---

## 9. Performance Considerations

### 9.1 Current Implementation

The current implementation prioritizes **correctness over performance**:

- Sequential triangular solve (single thread)
- Naive GEMM implementations
- No kernel fusion beyond attention matrix
- Minimal register optimization

### 9.2 Memory Access Patterns

| Operation | Pattern | Notes |
|-----------|---------|-------|
| Element-wise | Coalesced | Optimal |
| GEMM naive | Strided B access | Poor for NT |
| GEMM tiled | LDS staging | Good |
| Transpose | Strided write | Could use LDS |
| Broadcast mul | Coalesced + broadcast | Acceptable |

### 9.3 Occupancy Analysis

| Kernel | Block Size | Registers | LDS | Occupancy |
|--------|------------|-----------|-----|-----------|
| Element-wise | 256×1 | ~8 | 0 | High |
| GEMM naive | 16×16 | ~16 | 0 | Medium |
| GEMM tiled | 16×16 | ~20 | 2KB | Medium |
| Solve tri | 1×1 | ~8 | 0 | Very Low |

---

## 10. Future Optimization Opportunities

### 10.1 Triangular Solve Parallelization

The sequential triangular solve is the main bottleneck. Options:

1. **Wavefront parallelism**: Solve diagonal wavefronts in parallel
2. **Recursive blocking**: Block triangular solve with GEMM updates
3. **Approximate methods**: Iterative refinement

### 10.2 Kernel Fusion

Fuse consecutive operations to reduce memory traffic:

```
Fused kernel candidates:
- sigmoid + broadcast_mul (Beta processing)
- cumsum + exp + broadcast_mul (gate processing)
- GEMM + elementwise (attention matrix)
- scale + GEMM + add (state update)
```

### 10.3 Memory Layout Optimization

- **Packed state**: Use [S/4, S, 4] layout for vectorized loads
- **Interleaved Q/K/V**: Better cache utilization
- **Double buffering**: Hide memory latency in GEMM

### 10.4 LDS Optimization

- **Bank conflict avoidance**: Pad shared memory arrays
- **Vectorized LDS access**: Use float4 loads
- **Persistent threads**: Keep data in LDS across operations

### 10.5 Full Fusion Target

Ultimate goal: Single kernel for entire intra-chunk computation

```
Requirements:
- ~40KB LDS for matrices
- Register blocking for GEMM
- Wavefront-parallel triangular solve
- Careful synchronization
```

---

## Appendix A: File Structure

```
experiments/deltanet/
├── Makefile              # Build system
├── reference.h           # Host reference API
├── reference.cpp         # Host reference implementations
├── test_harness.cpp      # GPU kernels and tests
├── test_deltanet         # Built executable
└── DELTANET_KERNEL_DOCUMENTATION.md  # This document
```

---

## Appendix B: Test Results Summary

```
========================================
Test Summary: 26/26 passed
========================================

Phase 1: Element-wise Operations
  [PASS] sigmoid     (max_diff=1.19e-07)
  [PASS] exp         (max_diff=1.91e-06)
  [PASS] mul         (max_diff=0.00e+00)
  [PASS] cumsum      (max_diff=0.00e+00)

Phase 2: Triangular Operations
  [PASS] tril        (max_diff=0.00e+00)
  [PASS] causal_mask (max_diff=0.00e+00)
  [PASS] eye         (max_diff=0.00e+00)
  [PASS] solve_tri_small (max_diff=0.00e+00)
  [PASS] solve_tri_64    (max_diff=2.38e-07)

Phase 3: Matrix Operations
  [PASS] gemm_nt_naive_small (max_diff=1.19e-07)
  [PASS] gemm_nt_naive_64    (max_diff=7.15e-07)
  [PASS] gemm_tn_naive       (max_diff=7.15e-07)
  [PASS] gemm_nt_tiled       (max_diff=7.15e-07)
  [PASS] gemm_nt_tiled_128   (max_diff=4.77e-07)

Phase 4: Decay Mask
  [PASS] decay_mask_small (max_diff=1.19e-07)
  [PASS] decay_mask_64    (max_diff=2.38e-07)

Phase 5: Attention Matrix
  [PASS] attn_matrix_small (max_diff=5.96e-08)
  [PASS] attn_matrix_64    (max_diff=3.58e-07)

Phase 6: Intra-Chunk
  [PASS] attn_pre_computation    (max_diff=7.45e-09)
  [PASS] solve_with_causal_identity (max_diff=7.45e-09)
  [PASS] intra_chunk_small (max_diff=2.98e-08)
  [PASS] intra_chunk       (max_diff=1.19e-07)

Phase 7: State Interaction
  [PASS] state_update      (max_diff=2.38e-07)
  [PASS] output_with_state (max_diff=2.38e-07)

Phase 8: Full Delta-Net
  [PASS] deltanet_single_chunk (max_diff=7.15e-07)
  [PASS] deltanet_full         (max_diff=0.00e+00)
```

---

## Appendix C: Common Issues and Solutions

### Issue: Transpose grid dimension bug

**Symptom**: Large max_diff (~1.0) in K_cumdecay computation

**Cause**: Grid dimensions swapped for transpose kernel

**Solution**: Use grid matching INPUT dimensions, not OUTPUT
```cpp
// WRONG: grid_SC for [S,C] output
kernel_transpose<<<grid_SC, block2d>>>(d_temp, d_K_cumdecay, C, S);

// CORRECT: grid_CS for [C,S] input
kernel_transpose<<<grid_CS, block2d>>>(d_temp, d_K_cumdecay, C, S);
```

### Issue: Causal mask only writing row 0

**Symptom**: Only first row has correct values

**Cause**: 2D kernel using `blockIdx.y` without `threadIdx.y`

**Solution**: Proper 2D indexing
```cpp
// WRONG
int row = blockIdx.y;

// CORRECT
int row = blockIdx.y * blockDim.y + threadIdx.y;
```

### Issue: No on-device debugging

**Symptom**: Can't use printf in kernels

**Solution**:
1. Copy intermediate results to host
2. Compare against reference
3. Binary search for divergence point

---

## References

1. Yang et al., "Parallelizing Linear Transformers with the Delta Rule over Sequence Length", NeurIPS 2024
2. Schlag et al., "Linear Transformers Are Secretly Fast Weight Programmers", ICML 2021
3. Qwen Team, "Qwen Technical Report", 2024
4. AMD ROCm Documentation, https://rocm.docs.amd.com/

---

*Document generated for Delta-Net kernel validation framework v1.0*
*Target: AMD MI50 (gfx906) with ROCm*
*Tests: 26/26 passing*
