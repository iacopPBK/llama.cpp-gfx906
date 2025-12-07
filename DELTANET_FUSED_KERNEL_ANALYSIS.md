# Delta-Net Fused Kernel Analysis for GFX906

**Goal:** Design a single fused HIP kernel for Delta-Net linear attention on AMD MI50 (gfx906)

---

## Part 1: Qwen3-Next Model Dimensions

### 1.1 Key Parameters
```
n_embd           = 2048    (hidden size)
n_layer          = 48      (total layers, 36 are Delta-Net)
ssm_d_conv       = 4       (convolution kernel size)
ssm_d_inner      = 4096    (inner dimension)
ssm_d_state      = 128     (state size = head_k_dim)
ssm_n_group      = 16      (groups = num_k_heads)
ssm_dt_rank      = 32      (time step rank = num_v_heads)
```

### 1.2 Derived Dimensions
```cpp
head_k_dim   = ssm_d_state = 128                      // Key head dimension
head_v_dim   = ssm_d_inner / ssm_dt_rank = 4096/32 = 128  // Value head dimension
num_k_heads  = ssm_n_group = 16                       // Number of K heads
num_v_heads  = ssm_dt_rank = 32                       // Number of V heads
S_k = S_v    = 128                                    // State dimensions (both 128)
CHUNK_SIZE   = 64                                     // Tokens per chunk
```

### 1.3 Critical Observation
**K heads (16) != V heads (32)** - K heads are repeated 2x to match V heads before Delta-Net computation.

---

## Part 2: Delta-Net Mathematical Algorithm

### 2.1 The Delta Rule (Core Insight)
Standard linear attention suffers from **memory overload** - it can only add associations, never erase them.

Delta-Net fixes this with a **gradient descent update**:
```
S_t = S_{t-1} - beta_t * (S_{t-1} @ k_t - v_t) @ k_t^T
```

This is equivalent to:
```
S_t = S_{t-1} @ (I - beta_t * k_t @ k_t^T) + beta_t * v_t @ k_t^T
```

Where:
- `S_t` is the state matrix (d x d)
- `beta_t` is the "write strength" (learned, sigmoid-ed to [0,1])
- `k_t, v_t` are key and value vectors

### 2.2 The WY Representation (Key to Parallelization)

The transition matrix `(I - beta * k @ k^T)` is a **generalized Householder transformation**.

Products of such matrices have a compact "WY representation":
```
prod_{i=1}^{t} (I - beta_i * k_i @ k_i^T) = I - sum_{i=1}^{t} w_i @ k_i^T
```

Where `w_i` is computed recursively:
```
w_t = beta_t * (k_t - sum_{i=1}^{t-1} w_i * (k_i^T @ k_t))
```

Similarly for state:
```
S_n = sum_{t=1}^{n} u_t @ k_t^T
```

Where:
```
u_t = beta_t * (v_t - sum_{i=1}^{t-1} u_i * (k_i^T @ k_t))
```

### 2.3 The UT Transform (Triangular Solve)

To compute W and U efficiently using matrix operations:

1. **Build adjacency matrix A:**
   ```
   A = tril(-diag(beta) @ K @ K^T, -1)  // Strictly lower triangular
   ```

2. **Solve triangular system:**
   ```
   T = (I - A)^{-1}  // Forward substitution
   ```

3. **Compute W and U:**
   ```
   W = T @ diag(beta) @ K
   U = T @ diag(beta) @ V
   ```

This is the `solve_tri` operation in llama.cpp!

---

## Part 3: Chunked Parallel Algorithm

### 3.1 Why Chunking?

- **Recurrent form**: O(L * d^2) - cannot parallelize across sequence
- **Fully parallel**: O(L^2 * d) - quadratic in sequence length
- **Chunked**: O(L * C * d + L * d^2 / C) - best of both worlds

With CHUNK_SIZE=64, we process sequences in 64-token chunks.

### 3.2 Intra-Chunk Computation (Parallel)

For each chunk [i], compute:

**Step 1: Build attention matrix**
```
k_beta = k * beta                    // [S_k, chunk, H, batch]
v_beta = v * beta                    // [S_v, chunk, H, batch]
kmulkbeta = k @ k_beta^T             // [chunk, chunk, H, batch] - GEMM!
```

**Step 2: Apply decay mask**
```
g_cumsum = cumsum(g)                 // Cumulative gate values
decay_mask = exp((g_cumsum_j - g_cumsum_i) * causal_diag_mask)
k_decay = kmulkbeta * decay_mask
attn = -k_decay * causal_mask
```

**Step 3: Triangular solve (WY representation)**
```
attn_lower = attn * causal_mask
lhs = I - attn_lower
attn_solved = solve_tri(lhs, attn)   // Forward substitution
attn = attn_solved * causal_mask + I
```

**Step 4: Compute transformed V and K**
```
v_new = attn @ v_beta^T              // GEMM!
kbeta_gexp = k_beta * exp(g_cumsum)
k_cumdecay = (attn @ kbeta_gexp^T)^T // GEMM!
```

### 3.3 Inter-Chunk Computation (Sequential per chunk)

For each chunk, interact with the recurrent state:

**Step 5: Query-state interaction**
```
attn = q_chunk @ k_chunk^T * decay_mask  // GEMM!
v_prime = k_cumdecay @ state^T           // GEMM! (state interaction)
v_new = v_chunk - v_prime
```

**Step 6: Compute output**
```
q_gexp = q_chunk * exp(g_cumsum)
attn_inter = q_gexp @ state^T            // GEMM! (state interaction)
v_attn = attn @ v_new^T                  // GEMM!
output_chunk = attn_inter + v_attn
```

**Step 7: Update state**
```
g_last = g_cumsum[:, -1]                 // Last gate value in chunk
g_diff = g_last - g_cumsum
key_gdiff = k_chunk * exp(g_diff)
kgdmulvnew = key_gdiff^T @ v_new         // GEMM!
state = state * exp(g_last) + kgdmulvnew
```

---

## Part 4: GEMM Explosion Analysis

### 4.1 GEMMs per Chunk (build_delta_net_chunking)

| Line | Operation | Dimensions | Purpose |
|------|-----------|------------|---------|
| 233 | `k @ k_beta^T` | [C,S_k] @ [S_k,C] = [C,C] | Attention base |
| 249 | `v_beta^T @ attn` | [S_v,C] @ [C,C] = [S_v,C] | Value transform |
| 259 | `attn @ kbeta_gexp^T` | [C,C] @ [C,S_k] = [C,S_k] | Cumulative decay |
| 292 | `k_chunk @ q_chunk^T` | [S_k,C] @ [C,S_k] = [S_k,S_k] | Q-K attention |
| 299 | `state^T @ k_cumdecay` | [S_v,S_v] @ [S_v,C] = [S_v,C] | State-key interaction |
| 307 | `state^T @ q_gexp` | [S_v,S_v] @ [S_v,C] = [S_v,C] | State-query interaction |
| 310 | `v_new^T @ attn` | [C,S_v] @ [C,C] = [S_v,C] | Value-attention |
| 343 | `v_new^T @ key_gdiff^T` | [C,S_v] @ [S_v,C] = [C,C] | State update |

**Per chunk: 8+ GEMMs**
**Per layer: 8 * n_chunks GEMMs**
**36 layers with pp=1024 (16 chunks): 36 * 8 * 16 = 4,608 GEMMs**

Plus out-of-loop GEMMs, MoE, and other operations = **~885,000 total GEMMs**

### 4.2 Why ISA000 Fallback?

The small matrix GEMMs trigger generic kernels:
```
[C, C] = [64, 64]           -> MT16x16x8 ISA000 (bad!)
[S_v, S_v] = [128, 128]     -> MT16x16x8 ISA000 (bad!)
[S_k, C] = [128, 64]        -> Sometimes MT64x64x16 ISA906 (good)
```

The 128x128 and 64x64 matrices are too small for efficient rocBLAS kernel selection.

---

## Part 5: Fused Kernel Design

### 5.1 What to Fuse

A fused Delta-Net kernel should combine:
1. **Decay mask computation** (cumsum, exp, mul)
2. **Attention matrix build** (k @ k_beta^T * decay_mask)
3. **Triangular solve** (forward substitution in-place)
4. **Value and key transforms** (matrix multiplies)
5. **State interaction** (the chunk loop body)
6. **State update** (exp and accumulate)

### 5.2 Memory Layout

Input tensors per head:
```
q:     [S_k=128, n_tokens, 1]     // Query
k:     [S_k=128, n_tokens, 1]     // Key
v:     [S_v=128, n_tokens, 1]     // Value
g:     [n_tokens, 1, 1]           // Gate
beta:  [1, n_tokens, 1]           // Write strength
state: [S_v=128, S_v=128, 1]      // Recurrent state (128x128 = 16KB per head!)
```

### 5.3 Shared Memory Requirements (gfx906: 64KB LDS)

Per chunk per head:
```
q_chunk:     128 * 64 * 4 = 32 KB
k_chunk:     128 * 64 * 4 = 32 KB
v_chunk:     128 * 64 * 4 = 32 KB
attn:        64 * 64 * 4  = 16 KB
decay_mask:  64 * 64 * 4  = 16 KB
state:       128 * 128 * 4 = 64 KB  <- This alone fills LDS!
```

**Problem**: State matrix alone is 64KB, equal to total LDS!

### 5.4 Tiling Strategy

We cannot fit everything in LDS. Must tile:

**Option A: Tile over heads**
- Process 1 head at a time
- State fits in LDS (64KB)
- Reload Q, K, V per chunk

**Option B: Tile over state dimensions**
- Split state into 4x4 = 16 tiles of [32, 32]
- Each tile is 4KB
- More complex indexing

**Option C: Hybrid chunked approach**
- Keep attention matrix (16KB) and key/query tiles in LDS
- Stream state through registers
- Use global memory for state updates

### 5.5 Register Usage (gfx906: 256 VGPRs/wave)

Key values to keep in registers:
```
accumulator:  4 * 4 = 16 VGPRs (4x4 tile output)
a_frag:       4 VGPRs (input tile A)
b_frag:       4 VGPRs (input tile B)
scalars:      8 VGPRs (beta, gate, masks)
```

Target: ~64 VGPRs for good occupancy (4 waves/SIMD)

### 5.6 Proposed Kernel Structure

```cpp
__global__ void deltanet_fused_kernel(
    const float* __restrict__ Q,      // [S_k, n_tokens, H, batch]
    const float* __restrict__ K,      // [S_k, n_tokens, H, batch]
    const float* __restrict__ V,      // [S_v, n_tokens, H, batch]
    const float* __restrict__ G,      // [n_tokens, H, batch]
    const float* __restrict__ Beta,   // [n_tokens, H, batch]
    float* __restrict__ State,        // [S_v, S_v, H, batch] (in/out)
    float* __restrict__ Output,       // [S_v, n_tokens, H, batch]
    int n_tokens, int n_chunks
) {
    // Each block handles one (head, batch) pair
    const int head_batch_idx = blockIdx.x;

    __shared__ float s_attn[64][64];       // 16 KB
    __shared__ float s_k_chunk[128][64];   // 32 KB
    __shared__ float s_q_chunk[128][64];   // 32 KB (shared with k in ping-pong)

    // State in registers (tiled 32x32)
    float state_tile[4][4];  // Each thread owns a 4x4 output tile

    // Load initial state tile from global memory
    load_state_tile(State, state_tile, head_batch_idx);

    for (int chunk = 0; chunk < n_chunks; chunk++) {
        // 1. Load Q, K, V, G, Beta for this chunk into LDS
        load_chunk_to_lds(Q, K, V, G, Beta, chunk, ...);
        __syncthreads();

        // 2. Compute decay_mask = exp(cumsum(g) * causal_mask)
        compute_decay_mask_inplace(s_attn, G + chunk*64);

        // 3. Compute attn = -(k @ k_beta^T * decay_mask) * causal
        compute_attention_matrix(s_k_chunk, s_attn);

        // 4. Triangular solve: attn = (I - attn)^{-1} @ attn
        triangular_solve_inplace(s_attn);  // Forward substitution

        // 5. State interaction (the expensive part)
        // v_prime = k_cumdecay @ state
        // output = q_gexp @ state + attn @ v_new
        compute_output_with_state(s_q_chunk, s_k_chunk, s_attn,
                                   state_tile, Output + chunk*64);

        // 6. Update state: state = state * exp(g_last) + k @ v_new
        update_state_tile(state_tile, s_k_chunk, ...);
        __syncthreads();
    }

    // Write final state back
    store_state_tile(State, state_tile, head_batch_idx);
}
```

---

## Part 6: GFX906-Specific Optimizations

### 6.1 Available Instructions

```
v_fma_f32           - Fused multiply-add (use everywhere!)
v_exp_f32           - Exponential (hardware accelerated on gfx906)
ds_read_b128        - 128-bit LDS load (4 floats)
ds_write_b128       - 128-bit LDS write
global_load_dwordx4 - 128-bit global load
v_mac_f32           - Multiply-accumulate
```

### 6.2 Wavefront Considerations

- gfx906 wavefront = 64 threads
- 64x64 attention matrix = 4096 elements = 64 elements/thread
- Each thread processes one row of attention matrix
- Natural mapping!

### 6.3 Triangular Solve Strategy

Forward substitution can be done efficiently:
```cpp
// Each thread handles one row of the 64x64 matrix
for (int col = 0; col < row; col++) {
    // Broadcast solved value from thread 'col'
    float solved = __shfl(x[col], col);
    x[row] -= s_attn[row][col] * solved;
}
x[row] /= s_attn[row][row];  // Diagonal
```

This uses **wave shuffles** for efficient broadcast - no LDS bank conflicts!

### 6.4 State Tiling

Since state is 128x128 and doesn't fit in LDS:
```
Option 1: Process in 4 passes of 32x128 tiles (32KB each)
Option 2: Process in 16 passes of 32x32 tiles (4KB each)
Option 3: Keep 64x64 center in LDS, stream edges from global
```

---

## Part 7: Expected Performance

### 7.1 Current Performance (885K separate GEMMs)

- Total GPU time: 6,074 ms
- ISA000 time: 1,413 ms (23%)
- Launch overhead: Significant

### 7.2 Target Performance (Fused Kernel)

- Eliminate 800K+ kernel launches
- No ISA000 fallback (custom kernel is ISA906)
- Better memory bandwidth utilization
- Expected speedup: **2-4x** on Delta-Net layers

### 7.3 Bottleneck Analysis

The fused kernel will be:
- **Compute bound** for attention matrix (O(C^2 * d) = O(64^2 * 128))
- **Memory bound** for state updates (128*128 = 16K floats per head per chunk)

With 32 V heads per layer, state bandwidth = 32 * 16K * 4 bytes * 2 (read+write) = 4MB per layer per chunk.

---

## Part 8: Implementation Roadmap

### Phase 1: Prototype (Week 1)
1. Implement single-head fused kernel
2. Test correctness against reference
3. Basic performance measurement

### Phase 2: Optimize (Week 2)
1. Add LDS tiling for state
2. Implement wave shuffle triangular solve
3. Tune occupancy and register usage

### Phase 3: Production (Week 3)
1. Multi-head batching
2. Edge case handling (variable sequence lengths)
3. Integration with llama.cpp

---

## Appendix A: Key Code Locations

| Component | File | Line |
|-----------|------|------|
| Delta-Net chunking | src/models/qwen3next.cpp | 87-362 |
| Delta-Net recurrent | src/models/qwen3next.cpp | 364-611 |
| Triangular mask | ggml/src/ggml-cuda/tri.cu | 7-136 |
| Triangular solve | ggml/src/ggml-cuda/solve_tri.cu | 1-203 |
| rocBLAS SGEMM bypass | ggml/src/ggml-cuda/ggml-cuda.cu | 1359+ |

---

## Appendix B: References

1. [DeltaNet Paper](https://arxiv.org/abs/2406.06484) - "Parallelizing Linear Transformers with the Delta Rule over Sequence Length"
2. [DeltaNet Explained Blog](https://sustcsonglin.github.io/blog/2024/deltanet-1/) - Mathematical foundations
3. [Flash-Linear-Attention](https://github.com/fla-org/flash-linear-attention) - Reference Triton implementation
4. [Gated Delta Networks](https://openreview.net/forum?id=r8H7xhYPwz) - ICLR 2025 extension

---

*Analysis by Claude Code - December 7, 2025*
