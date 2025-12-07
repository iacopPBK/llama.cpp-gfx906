# Delta-Net Fused Kernel: Incremental Implementation Plan

**Target:** AMD MI50 (gfx906) - No on-device printf!
**Validation Strategy:** Host-side comparison with reference implementation

---

## Phase 0: Test Infrastructure (CRITICAL FIRST STEP)

Before writing ANY kernel code, we need a test harness.

### Step 0.1: Create Test Harness

```cpp
// tests/test_deltanet_kernel.hip
// Standalone test that can run WITHOUT llama.cpp

#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>

// Test dimensions (match Qwen3-Next)
constexpr int S_K = 128;        // Key/Query head dimension
constexpr int S_V = 128;        // Value head dimension
constexpr int CHUNK = 64;       // Chunk size
constexpr int N_HEADS = 1;      // Start with 1 head
constexpr float EPS = 1e-4f;    // Tolerance for float comparison

// Host reference implementation (GOLDEN)
void host_reference_deltanet(...);

// Device kernel under test
__global__ void deltanet_kernel_under_test(...);

// Comparison utility
bool compare_tensors(const float* a, const float* b, int n, float eps) {
    float max_diff = 0;
    int max_idx = 0;
    for (int i = 0; i < n; i++) {
        float diff = std::abs(a[i] - b[i]);
        if (diff > max_diff) {
            max_diff = diff;
            max_idx = i;
        }
    }
    std::cout << "Max diff: " << max_diff << " at index " << max_idx
              << " (ref=" << a[max_idx] << ", got=" << b[max_idx] << ")\n";
    return max_diff < eps;
}

int main() {
    // Allocate host memory
    // Allocate device memory
    // Initialize with known test data
    // Run host reference
    // Run device kernel
    // Copy back and compare
    // Report PASS/FAIL
}
```

**Validation:** Compiles and runs, even if kernel is empty.

### Step 0.2: Create Reference Implementation on Host

Write a **pure C++ host implementation** of each Delta-Net sub-operation. This becomes our golden reference.

```cpp
// Reference: Triangular mask extraction
void host_tri_lower(const float* src, float* dst, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            dst[i * n + j] = (j < i) ? src[i * n + j] : 0.0f;
        }
    }
}

// Reference: Forward substitution (triangular solve)
void host_solve_tri(const float* A, const float* B, float* X, int n, int k) {
    // A is n x n lower triangular, B is n x k, solve AX = B
    for (int col = 0; col < k; col++) {
        for (int row = 0; row < n; row++) {
            float sum = B[row * k + col];
            for (int j = 0; j < row; j++) {
                sum -= A[row * n + j] * X[j * k + col];
            }
            X[row * k + col] = sum / A[row * n + row];
        }
    }
}

// Reference: Cumulative sum
void host_cumsum(const float* src, float* dst, int n) {
    float sum = 0;
    for (int i = 0; i < n; i++) {
        sum += src[i];
        dst[i] = sum;
    }
}

// Reference: Decay mask computation
void host_decay_mask(const float* g_cumsum, float* mask, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (j <= i) {
                mask[i * n + j] = expf(g_cumsum[j] - g_cumsum[i]);
            } else {
                mask[i * n + j] = 0.0f;
            }
        }
    }
}

// Reference: Matrix multiply (naive)
void host_matmul(const float* A, const float* B, float* C,
                 int M, int N, int K, bool transA, bool transB) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0;
            for (int k = 0; k < K; k++) {
                float a = transA ? A[k * M + i] : A[i * K + k];
                float b = transB ? B[j * K + k] : B[k * N + j];
                sum += a * b;
            }
            C[i * N + j] = sum;
        }
    }
}
```

**Validation:** Each host function tested independently with known inputs.

---

## Phase 1: Element-wise Operations

Start with the simplest kernels that don't require LDS or complex synchronization.

### Step 1.1: Sigmoid Kernel

```cpp
__global__ void kernel_sigmoid(const float* __restrict__ in,
                                float* __restrict__ out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = 1.0f / (1.0f + expf(-in[idx]));
    }
}
```

**Test:**
```cpp
void test_sigmoid() {
    std::vector<float> h_in = {-2, -1, 0, 1, 2};
    std::vector<float> h_ref(5), h_out(5);

    // Host reference
    for (int i = 0; i < 5; i++)
        h_ref[i] = 1.0f / (1.0f + expf(-h_in[i]));

    // Device
    float *d_in, *d_out;
    hipMalloc(&d_in, 5 * sizeof(float));
    hipMalloc(&d_out, 5 * sizeof(float));
    hipMemcpy(d_in, h_in.data(), 5 * sizeof(float), hipMemcpyHostToDevice);

    kernel_sigmoid<<<1, 256>>>(d_in, d_out, 5);

    hipMemcpy(h_out.data(), d_out, 5 * sizeof(float), hipMemcpyDeviceToHost);

    assert(compare_tensors(h_ref.data(), h_out.data(), 5, 1e-5f));
    std::cout << "PASS: sigmoid\n";
}
```

**Validation:** Output matches host reference within tolerance.

### Step 1.2: Exponential Kernel

```cpp
__global__ void kernel_exp(const float* __restrict__ in,
                           float* __restrict__ out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = expf(in[idx]);
    }
}
```

**Validation:** Same pattern as sigmoid.

### Step 1.3: Element-wise Multiply

```cpp
__global__ void kernel_mul(const float* __restrict__ a,
                           const float* __restrict__ b,
                           float* __restrict__ c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] * b[idx];
    }
}
```

**Validation:** c[i] == a[i] * b[i] for all i.

### Step 1.4: Cumulative Sum (Parallel Prefix Sum)

This is trickier - requires proper parallel algorithm.

```cpp
// Simple sequential version first (single block, single thread)
__global__ void kernel_cumsum_sequential(const float* __restrict__ in,
                                          float* __restrict__ out, int n) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float sum = 0;
        for (int i = 0; i < n; i++) {
            sum += in[i];
            out[i] = sum;
        }
    }
}
```

**Validation:** Match host_cumsum output exactly.

---

## Phase 2: Triangular Operations (Core of Delta-Net!)

### Step 2.1: Triangular Mask Extraction

```cpp
// Extract lower triangular part of matrix (zero out upper)
__global__ void kernel_tril(const float* __restrict__ src,
                            float* __restrict__ dst,
                            int n, int diag_offset) {
    int row = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        int idx = row * n + col;
        // Lower triangular: col <= row + diag_offset
        dst[idx] = (col <= row + diag_offset) ? src[idx] : 0.0f;
    }
}
```

**Test data:**
```
Input:              Expected (diag=0):
[1 2 3]             [1 0 0]
[4 5 6]    ->       [4 5 0]
[7 8 9]             [7 8 9]
```

**Validation:** Compare with host_tri_lower.

### Step 2.2: Causal + Identity Mask

```cpp
// Create (causal_mask + identity) matrix
__global__ void kernel_causal_diag_mask(float* __restrict__ out, int n) {
    int row = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        // 1.0 on diagonal and below, 0.0 above
        out[row * n + col] = (col <= row) ? 1.0f : 0.0f;
    }
}
```

**Validation:** Visual inspection + host comparison for small matrices.

### Step 2.3: Triangular Solve - Sequential Version First!

Start with a SIMPLE sequential version to verify correctness:

```cpp
// Sequential forward substitution (SLOW but correct)
// Solves (I - A) * X = B where A is strictly lower triangular
__global__ void kernel_solve_tri_sequential(
    const float* __restrict__ A,  // n x n, strictly lower triangular
    const float* __restrict__ B,  // n x k
    float* __restrict__ X,        // n x k output
    int n, int k
) {
    // Single thread does all the work (SLOW!)
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        for (int col = 0; col < k; col++) {
            for (int row = 0; row < n; row++) {
                float sum = B[row * k + col];
                for (int j = 0; j < row; j++) {
                    // (I - A)[row][j] = -A[row][j] for j < row
                    // (I - A)[row][row] = 1
                    sum += A[row * n + j] * X[j * k + col];  // Note: + because (I-A)
                }
                X[row * k + col] = sum;  // Diagonal of (I-A) is 1
            }
        }
    }
}
```

**Test case:**
```
A (strictly lower tri):    B:          Expected X:
[0   0   0]               [1]         Solve (I-A)X = B
[0.5 0   0]               [2]         [1    0    0  ][x0]   [1]
[0.3 0.2 0]               [3]         [-0.5 1    0  ][x1] = [2]
                                      [-0.3 -0.2 1  ][x2]   [3]

x0 = 1
x1 = 2 + 0.5*1 = 2.5
x2 = 3 + 0.3*1 + 0.2*2.5 = 3.8
```

**Validation:** CRITICAL - this must match host_solve_tri exactly!

### Step 2.4: Triangular Solve - Wave Shuffle Version

Once sequential version passes, implement the fast version:

```cpp
// Parallel forward substitution using wave shuffles
// Each thread handles one row of the 64x64 matrix
__global__ void kernel_solve_tri_wave(
    const float* __restrict__ A,  // 64 x 64
    float* __restrict__ X,        // 64 x 1 (in-place B -> X)
    int n  // Must be <= 64 for single wavefront
) {
    int row = threadIdx.x;
    if (row >= n) return;

    float x_row = X[row];  // Load my row of B

    // Forward substitution with wave shuffles
    for (int col = 0; col < row; col++) {
        // Get the solved value from thread 'col'
        float x_col = __shfl(x_row, col);  // Actually need solved value!
        // This is tricky - need to propagate solutions...
    }

    // Actually, simpler approach - iterative:
    for (int iter = 0; iter < n; iter++) {
        if (row == iter) {
            // I'm the current row being solved
            // x_row is already correct (accumulated from previous iters)
        }
        // Broadcast the solved value for row 'iter'
        float solved = __shfl(x_row, iter);
        if (row > iter) {
            x_row += A[row * n + iter] * solved;
        }
        __syncthreads();  // Ensure all threads see the update
    }

    X[row] = x_row;
}
```

**Validation:** Must produce IDENTICAL output to sequential version.

---

## Phase 3: Matrix Multiply (Small Matrices)

### Step 3.1: Naive 64x64 GEMM

```cpp
// C = A @ B^T where A is [M, K], B is [N, K]
__global__ void kernel_gemm_naive(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[col * K + k];  // B transposed
        }
        C[row * N + col] = sum;
    }
}
```

**Validation:** Compare with host_matmul.

### Step 3.2: LDS-Tiled GEMM

```cpp
#define TILE 16

__global__ void kernel_gemm_tiled(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    __shared__ float s_A[TILE][TILE];
    __shared__ float s_B[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float sum = 0;

    for (int t = 0; t < (K + TILE - 1) / TILE; t++) {
        // Load tiles to LDS
        int a_col = t * TILE + threadIdx.x;
        int b_row = t * TILE + threadIdx.y;

        s_A[threadIdx.y][threadIdx.x] = (row < M && a_col < K) ?
                                         A[row * K + a_col] : 0.0f;
        s_B[threadIdx.y][threadIdx.x] = (b_row < K && col < N) ?
                                         B[col * K + b_row] : 0.0f;
        __syncthreads();

        // Compute
        for (int k = 0; k < TILE; k++) {
            sum += s_A[threadIdx.y][k] * s_B[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}
```

**Validation:** Same output as naive version, but faster.

---

## Phase 4: Decay Mask Computation

### Step 4.1: Decay Mask from Cumsum

```cpp
// decay_mask[i][j] = exp(g_cumsum[j] - g_cumsum[i]) * causal[i][j]
__global__ void kernel_decay_mask(
    const float* __restrict__ g_cumsum,  // [n]
    float* __restrict__ mask,            // [n, n]
    int n
) {
    int row = blockIdx.y;
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

**Validation:** Compare with host_decay_mask.

---

## Phase 5: Attention Matrix Construction

### Step 5.1: K @ K_beta^T * decay_mask

Combine previous pieces:

```cpp
__global__ void kernel_attention_matrix(
    const float* __restrict__ K,         // [S_k, C]
    const float* __restrict__ K_beta,    // [S_k, C]
    const float* __restrict__ decay_mask,// [C, C]
    float* __restrict__ attn,            // [C, C]
    int S_k, int C
) {
    // Step 1: Compute K @ K_beta^T into attn
    // Step 2: Multiply by decay_mask element-wise
    // Step 3: Negate and apply causal mask

    int row = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < C && col < C) {
        // K @ K_beta^T
        float sum = 0;
        for (int k = 0; k < S_k; k++) {
            sum += K[k * C + row] * K_beta[k * C + col];
        }

        // Apply decay and causal mask
        if (col < row) {  // Strictly lower triangular
            attn[row * C + col] = -sum * decay_mask[row * C + col];
        } else {
            attn[row * C + col] = 0.0f;
        }
    }
}
```

**Validation:** Compare each intermediate step with host reference.

---

## Phase 6: Integration - Intra-Chunk Computation

### Step 6.1: Complete Intra-Chunk Kernel

```cpp
__global__ void kernel_deltanet_intra_chunk(
    const float* __restrict__ Q,      // [S_k, C]
    const float* __restrict__ K,      // [S_k, C]
    const float* __restrict__ V,      // [S_v, C]
    const float* __restrict__ G,      // [C]
    const float* __restrict__ Beta,   // [C]
    float* __restrict__ attn_out,     // [C, C] - attention matrix after solve
    float* __restrict__ V_new,        // [S_v, C] - transformed values
    float* __restrict__ K_cumdecay,   // [S_k, C] - for state interaction
    int S_k, int S_v, int C
) {
    // This kernel computes everything WITHIN a chunk
    // State interaction happens in a separate kernel

    __shared__ float s_g_cumsum[CHUNK];
    __shared__ float s_beta[CHUNK];
    __shared__ float s_attn[CHUNK][CHUNK];
    __shared__ float s_decay[CHUNK][CHUNK];

    // Step 1: Load G and Beta, compute cumsum
    // Step 2: Compute decay mask
    // Step 3: Compute K @ K_beta^T * decay_mask
    // Step 4: Triangular solve
    // Step 5: Compute V_new = attn @ V_beta^T
    // Step 6: Compute K_cumdecay
}
```

**Validation:** Run the complete intra-chunk, compare all outputs.

---

## Phase 7: State Interaction (The Hard Part)

### Step 7.1: State @ Key Interaction

```cpp
// v_prime = K_cumdecay @ State^T
// State is [S_v, S_v] = [128, 128] - TOO BIG FOR LDS!

__global__ void kernel_state_key_interaction(
    const float* __restrict__ K_cumdecay,  // [S_k, C]
    const float* __restrict__ State,        // [S_v, S_v]
    float* __restrict__ V_prime,            // [S_v, C]
    int S_k, int S_v, int C
) {
    // Tile over State since it doesn't fit in LDS
    // Process 32x32 tiles of State at a time

    __shared__ float s_state_tile[32][32];
    __shared__ float s_k_tile[32][C];  // May need to tile this too

    // ...
}
```

**Validation:** Compare with host matmul of same inputs.

### Step 7.2: Output Computation

```cpp
// output = Q_gexp @ State^T + attn @ V_new
```

### Step 7.3: State Update

```cpp
// state_new = state * exp(g_last) + K @ V_new^T
```

---

## Phase 8: Full Fused Kernel

### Step 8.1: Single-Head, Single-Chunk

Combine all pieces into one kernel, test with 1 head, 1 chunk.

### Step 8.2: Single-Head, Multi-Chunk

Add the chunk loop.

### Step 8.3: Multi-Head

Add head batching.

---

## Debugging Strategy (NO PRINTF!)

### Method 1: Staged Output Buffers

```cpp
struct DebugBuffers {
    float* stage1_cumsum;      // After cumsum
    float* stage2_decay_mask;  // After decay mask
    float* stage3_attn_pre;    // Before triangular solve
    float* stage4_attn_post;   // After triangular solve
    float* stage5_v_new;       // After V transform
    // ... etc
};

__global__ void kernel_with_debug(
    /* normal inputs */,
    DebugBuffers* debug
) {
    // After each stage, write intermediate to debug buffer
    if (debug) {
        // Copy s_cumsum to debug->stage1_cumsum
    }
}
```

**Usage:** Run kernel, copy debug buffers to host, compare with reference at each stage.

### Method 2: Stage Flags

```cpp
__global__ void kernel_staged(
    /* inputs */,
    float* output,
    int max_stage  // Stop after this stage
) {
    // Stage 1
    compute_cumsum();
    if (max_stage == 1) { copy_to_output(); return; }

    // Stage 2
    compute_decay_mask();
    if (max_stage == 2) { copy_to_output(); return; }

    // ... etc
}
```

**Usage:** Run with max_stage=1, validate. Then max_stage=2, validate. Etc.

### Method 3: Reference Comparison Script

```python
#!/usr/bin/env python3
# validate_kernel.py

import numpy as np
import subprocess

def run_test(stage):
    # Run C++ test binary
    result = subprocess.run(['./test_deltanet', str(stage)],
                          capture_output=True, text=True)

    # Parse output (test binary dumps arrays to files)
    ref = np.fromfile(f'ref_stage{stage}.bin', dtype=np.float32)
    got = np.fromfile(f'got_stage{stage}.bin', dtype=np.float32)

    diff = np.abs(ref - got)
    print(f"Stage {stage}: max_diff={diff.max():.6e}, mean_diff={diff.mean():.6e}")

    if diff.max() > 1e-4:
        # Find worst elements
        worst = np.argsort(diff)[-10:]
        for i in worst:
            print(f"  [{i}]: ref={ref[i]:.6f}, got={got[i]:.6f}, diff={diff[i]:.6e}")
        return False
    return True

for stage in range(1, 10):
    if not run_test(stage):
        print(f"FAILED at stage {stage}")
        break
else:
    print("ALL STAGES PASSED!")
```

---

## File Structure

```
/home/iacoppbk/Desktop/llamacpp-gfx906/
├── experiments/
│   └── deltanet/
│       ├── Makefile
│       ├── test_harness.cpp        # Main test driver
│       ├── reference.cpp           # Host reference implementations
│       ├── reference.h
│       ├── kernels/
│       │   ├── elementwise.hip     # Phase 1 kernels
│       │   ├── triangular.hip      # Phase 2 kernels
│       │   ├── gemm.hip            # Phase 3 kernels
│       │   ├── decay.hip           # Phase 4 kernels
│       │   ├── attention.hip       # Phase 5 kernels
│       │   ├── intra_chunk.hip     # Phase 6 kernels
│       │   ├── state.hip           # Phase 7 kernels
│       │   └── fused.hip           # Phase 8 final kernel
│       ├── validate.py             # Python validation script
│       └── test_data/              # Known test vectors
│           ├── input_q.bin
│           ├── input_k.bin
│           ├── expected_attn.bin
│           └── ...
```

---

## Makefile

```makefile
HIPCC = /opt/rocm/bin/hipcc
HIPFLAGS = --offload-arch=gfx906 -O2 -g -DNDEBUG

all: test_deltanet

test_deltanet: test_harness.cpp reference.cpp kernels/*.hip
	$(HIPCC) $(HIPFLAGS) -o $@ test_harness.cpp reference.cpp

test_phase1: test_deltanet
	./test_deltanet --phase 1

test_phase2: test_deltanet
	./test_deltanet --phase 2

test_all: test_deltanet
	./test_deltanet --all

clean:
	rm -f test_deltanet *.bin
```

---

## Timeline

| Phase | Description | Complexity | Est. Time |
|-------|-------------|------------|-----------|
| 0 | Test infrastructure | Low | 2-3 hours |
| 1 | Element-wise ops | Low | 1-2 hours |
| 2 | Triangular ops | Medium | 4-6 hours |
| 3 | Small GEMM | Medium | 2-3 hours |
| 4 | Decay mask | Low | 1 hour |
| 5 | Attention matrix | Medium | 2-3 hours |
| 6 | Intra-chunk | High | 4-6 hours |
| 7 | State interaction | High | 6-8 hours |
| 8 | Full fusion | High | 4-6 hours |
| **Total** | | | **~30 hours** |

---

## Critical Success Factors

1. **Test infrastructure FIRST** - Don't write kernel code until you can validate it
2. **One piece at a time** - Each kernel must pass before moving to next
3. **Host reference is GOLDEN** - Any mismatch means kernel is wrong
4. **Small test cases first** - 4x4 matrices before 64x64
5. **Binary dumps for debugging** - Write intermediate results to files
6. **Bit-exact validation where possible** - Use fixed test inputs

---

## Ready to Start?

Begin with Phase 0, Step 0.1: Create the test harness.

