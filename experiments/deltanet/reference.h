#pragma once
/**
 * Delta-Net Host Reference Implementations
 *
 * These are the GOLDEN reference implementations for validating GPU kernels.
 * All GPU kernel outputs must match these within tolerance.
 *
 * Naming convention: host_<operation>
 */

#include <cstdint>

namespace deltanet {

// ============================================================================
// Constants (match Qwen3-Next dimensions)
// ============================================================================
constexpr int S_K = 128;          // Key head dimension (ssm_d_state)
constexpr int S_V = 128;          // Value head dimension
constexpr int CHUNK_SIZE = 64;    // Tokens per chunk
constexpr int NUM_K_HEADS = 16;   // Number of K heads (ssm_n_group)
constexpr int NUM_V_HEADS = 32;   // Number of V heads (ssm_dt_rank)

// ============================================================================
// Phase 1: Element-wise Operations
// ============================================================================

// Sigmoid: out[i] = 1 / (1 + exp(-in[i]))
void host_sigmoid(const float* in, float* out, int n);

// Exponential: out[i] = exp(in[i])
void host_exp(const float* in, float* out, int n);

// Element-wise multiply: out[i] = a[i] * b[i]
void host_mul(const float* a, const float* b, float* out, int n);

// Element-wise add: out[i] = a[i] + b[i]
void host_add(const float* a, const float* b, float* out, int n);

// Element-wise subtract: out[i] = a[i] - b[i]
void host_sub(const float* a, const float* b, float* out, int n);

// Element-wise negate: out[i] = -in[i]
void host_neg(const float* in, float* out, int n);

// Scale: out[i] = in[i] * scale
void host_scale(const float* in, float* out, int n, float scale);

// Cumulative sum: out[i] = sum(in[0..i])
void host_cumsum(const float* in, float* out, int n);

// ============================================================================
// Phase 2: Triangular Operations
// ============================================================================

// Extract lower triangular (zero out upper triangle)
// diag_offset: 0 = include diagonal, -1 = exclude diagonal
void host_tril(const float* src, float* dst, int n, int diag_offset = 0);

// Extract upper triangular (zero out lower triangle)
void host_triu(const float* src, float* dst, int n, int diag_offset = 0);

// Create identity matrix
void host_eye(float* dst, int n);

// Create causal mask (lower triangular of ones)
void host_causal_mask(float* dst, int n);

// Create causal + diagonal mask
void host_causal_diag_mask(float* dst, int n);

/**
 * Forward substitution: solve (I - A) * X = B
 *
 * A: n x n strictly lower triangular matrix (diagonal is ignored)
 * B: n x k right-hand side matrix
 * X: n x k solution matrix (output)
 *
 * The system (I - A) is unit lower triangular (ones on diagonal).
 * This is the core of Delta-Net's WY representation!
 */
void host_solve_tri(const float* A, const float* B, float* X, int n, int k);

/**
 * Solve triangular system with single RHS vector
 * Convenience wrapper for k=1 case
 */
void host_solve_tri_vec(const float* A, const float* b, float* x, int n);

// ============================================================================
// Phase 3: Matrix Operations
// ============================================================================

/**
 * General matrix multiply: C = alpha * op(A) * op(B) + beta * C
 *
 * trans_a: if true, use A^T
 * trans_b: if true, use B^T
 *
 * Dimensions:
 *   A: M x K (or K x M if transposed)
 *   B: K x N (or N x K if transposed)
 *   C: M x N
 */
void host_gemm(const float* A, const float* B, float* C,
               int M, int N, int K,
               bool trans_a, bool trans_b,
               float alpha = 1.0f, float beta = 0.0f);

// Simplified wrappers
void host_matmul(const float* A, const float* B, float* C, int M, int N, int K);
void host_matmul_tn(const float* A, const float* B, float* C, int M, int N, int K); // A^T @ B
void host_matmul_nt(const float* A, const float* B, float* C, int M, int N, int K); // A @ B^T
void host_matmul_tt(const float* A, const float* B, float* C, int M, int N, int K); // A^T @ B^T

// Matrix transpose
void host_transpose(const float* in, float* out, int rows, int cols);

// ============================================================================
// Phase 4: Decay Mask Computation
// ============================================================================

/**
 * Compute decay mask for Delta-Net attention
 *
 * decay_mask[i][j] = exp(g_cumsum[j] - g_cumsum[i]) if j <= i, else 0
 *
 * This creates the causal exponential decay pattern.
 */
void host_decay_mask(const float* g_cumsum, float* mask, int n);

/**
 * Compute decay mask with causal+diagonal multiplication
 *
 * Same as above but also multiplied by causal_diag_mask
 */
void host_decay_mask_causal(const float* g_cumsum, float* mask, int n);

// ============================================================================
// Phase 5: Attention Matrix Construction
// ============================================================================

/**
 * Build Delta-Net attention matrix (before triangular solve)
 *
 * attn = -(K @ K_beta^T) * decay_mask * causal_mask
 *
 * K:      [S_k, C] - keys
 * K_beta: [S_k, C] - keys * beta
 * decay:  [C, C]   - decay mask
 * attn:   [C, C]   - output attention matrix
 */
void host_attention_matrix(const float* K, const float* K_beta,
                           const float* decay_mask,
                           float* attn, int S_k, int C);

// ============================================================================
// Phase 6: Full Intra-Chunk Computation
// ============================================================================

/**
 * Complete intra-chunk Delta-Net computation
 *
 * Inputs:
 *   Q, K, V: [S, C] - query, key, value (S = S_k = S_v for simplicity)
 *   G:       [C]    - gate values
 *   Beta:    [C]    - write strength (will be sigmoided)
 *
 * Outputs:
 *   attn_solved: [C, C] - attention matrix after triangular solve
 *   V_new:       [S, C] - transformed values
 *   K_cumdecay:  [S, C] - keys with cumulative decay
 */
void host_intra_chunk(const float* Q, const float* K, const float* V,
                      const float* G, const float* Beta,
                      float* attn_solved, float* V_new, float* K_cumdecay,
                      int S, int C);

// ============================================================================
// Phase 7: State Interaction
// ============================================================================

/**
 * Compute output with state interaction
 *
 * output = Q_gexp @ State^T + attn @ V_new
 *
 * Q_gexp:  [S, C]    - query * exp(g_cumsum)
 * State:   [S, S]    - recurrent state matrix
 * attn:    [C, C]    - attention matrix
 * V_new:   [S, C]    - transformed values
 * output:  [S, C]    - output
 */
void host_output_with_state(const float* Q_gexp, const float* State,
                            const float* attn, const float* V_new,
                            float* output, int S, int C);

/**
 * Update recurrent state
 *
 * state_new = state * exp(g_last) + K_gdiff^T @ V_new
 */
void host_state_update(const float* State, const float* K_gdiff,
                       const float* V_new, float g_last,
                       float* State_new, int S, int C);

// ============================================================================
// Phase 8: Full Delta-Net (Multiple Chunks)
// ============================================================================

/**
 * Full Delta-Net chunked attention for a single head
 *
 * Q, K, V:    [S, n_tokens]
 * G:          [n_tokens]
 * Beta:       [n_tokens]
 * State_in:   [S, S]
 * Output:     [S, n_tokens]
 * State_out:  [S, S]
 */
void host_deltanet_full(const float* Q, const float* K, const float* V,
                        const float* G, const float* Beta,
                        const float* State_in,
                        float* Output, float* State_out,
                        int S, int n_tokens);

// ============================================================================
// Utility Functions
// ============================================================================

// Print matrix for debugging
void print_matrix(const float* m, int rows, int cols, const char* name);

// Compare two arrays, return max absolute difference
float max_diff(const float* a, const float* b, int n);

// Compare two arrays with tolerance, print errors
bool compare_arrays(const float* ref, const float* got, int n,
                    float eps = 1e-4f, const char* name = nullptr);

// Fill array with random values in [min, max]
void fill_random(float* arr, int n, float min_val = -1.0f, float max_val = 1.0f);

// Fill array with zeros
void fill_zeros(float* arr, int n);

// Fill array with ones
void fill_ones(float* arr, int n);

// Fill array with sequential values 0, 1, 2, ...
void fill_sequential(float* arr, int n);

} // namespace deltanet
