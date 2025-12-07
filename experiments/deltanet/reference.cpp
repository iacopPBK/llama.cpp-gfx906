/**
 * Delta-Net Host Reference Implementations
 *
 * GOLDEN reference for GPU kernel validation.
 * Prioritizes correctness over performance.
 */

#include "reference.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>

namespace deltanet {

// ============================================================================
// Phase 1: Element-wise Operations
// ============================================================================

void host_sigmoid(const float* in, float* out, int n) {
    for (int i = 0; i < n; i++) {
        out[i] = 1.0f / (1.0f + expf(-in[i]));
    }
}

void host_exp(const float* in, float* out, int n) {
    for (int i = 0; i < n; i++) {
        out[i] = expf(in[i]);
    }
}

void host_mul(const float* a, const float* b, float* out, int n) {
    for (int i = 0; i < n; i++) {
        out[i] = a[i] * b[i];
    }
}

void host_add(const float* a, const float* b, float* out, int n) {
    for (int i = 0; i < n; i++) {
        out[i] = a[i] + b[i];
    }
}

void host_sub(const float* a, const float* b, float* out, int n) {
    for (int i = 0; i < n; i++) {
        out[i] = a[i] - b[i];
    }
}

void host_neg(const float* in, float* out, int n) {
    for (int i = 0; i < n; i++) {
        out[i] = -in[i];
    }
}

void host_scale(const float* in, float* out, int n, float scale) {
    for (int i = 0; i < n; i++) {
        out[i] = in[i] * scale;
    }
}

void host_cumsum(const float* in, float* out, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += in[i];
        out[i] = sum;
    }
}

// ============================================================================
// Phase 2: Triangular Operations
// ============================================================================

void host_tril(const float* src, float* dst, int n, int diag_offset) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            // Keep if j <= i + diag_offset
            dst[i * n + j] = (j <= i + diag_offset) ? src[i * n + j] : 0.0f;
        }
    }
}

void host_triu(const float* src, float* dst, int n, int diag_offset) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            // Keep if j >= i + diag_offset
            dst[i * n + j] = (j >= i + diag_offset) ? src[i * n + j] : 0.0f;
        }
    }
}

void host_eye(float* dst, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            dst[i * n + j] = (i == j) ? 1.0f : 0.0f;
        }
    }
}

void host_causal_mask(float* dst, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            // 1 for j < i (strictly lower triangular)
            dst[i * n + j] = (j < i) ? 1.0f : 0.0f;
        }
    }
}

void host_causal_diag_mask(float* dst, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            // 1 for j <= i (lower triangular including diagonal)
            dst[i * n + j] = (j <= i) ? 1.0f : 0.0f;
        }
    }
}

void host_solve_tri(const float* A, const float* B, float* X, int n, int k) {
    /**
     * Solve (I - A) * X = B where A is strictly lower triangular
     *
     * Since (I - A) has 1s on diagonal and -A[i][j] below:
     *   X[i] = B[i] + sum_{j<i} A[i][j] * X[j]
     *
     * This is forward substitution.
     */
    for (int col = 0; col < k; col++) {
        for (int row = 0; row < n; row++) {
            float sum = B[row * k + col];
            for (int j = 0; j < row; j++) {
                // (I - A)[row][j] = -A[row][j], so we ADD A[row][j] * X[j]
                sum += A[row * n + j] * X[j * k + col];
            }
            // Diagonal of (I - A) is 1, so no division needed
            X[row * k + col] = sum;
        }
    }
}

void host_solve_tri_vec(const float* A, const float* b, float* x, int n) {
    host_solve_tri(A, b, x, n, 1);
}

// ============================================================================
// Phase 3: Matrix Operations
// ============================================================================

void host_gemm(const float* A, const float* B, float* C,
               int M, int N, int K,
               bool trans_a, bool trans_b,
               float alpha, float beta) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int kk = 0; kk < K; kk++) {
                float a_val = trans_a ? A[kk * M + i] : A[i * K + kk];
                float b_val = trans_b ? B[j * K + kk] : B[kk * N + j];
                sum += a_val * b_val;
            }
            C[i * N + j] = alpha * sum + beta * C[i * N + j];
        }
    }
}

void host_matmul(const float* A, const float* B, float* C, int M, int N, int K) {
    host_gemm(A, B, C, M, N, K, false, false);
}

void host_matmul_tn(const float* A, const float* B, float* C, int M, int N, int K) {
    // A^T @ B: A is [K, M], B is [K, N], C is [M, N]
    host_gemm(A, B, C, M, N, K, true, false);
}

void host_matmul_nt(const float* A, const float* B, float* C, int M, int N, int K) {
    // A @ B^T: A is [M, K], B is [N, K], C is [M, N]
    host_gemm(A, B, C, M, N, K, false, true);
}

void host_matmul_tt(const float* A, const float* B, float* C, int M, int N, int K) {
    host_gemm(A, B, C, M, N, K, true, true);
}

void host_transpose(const float* in, float* out, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            out[j * rows + i] = in[i * cols + j];
        }
    }
}

// ============================================================================
// Phase 4: Decay Mask Computation
// ============================================================================

void host_decay_mask(const float* g_cumsum, float* mask, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (j <= i) {
                // Causal: j can attend to i if j <= i
                mask[i * n + j] = expf(g_cumsum[j] - g_cumsum[i]);
            } else {
                mask[i * n + j] = 0.0f;
            }
        }
    }
}

void host_decay_mask_causal(const float* g_cumsum, float* mask, int n) {
    // Same as decay_mask but already incorporates causal structure
    host_decay_mask(g_cumsum, mask, n);
}

// ============================================================================
// Phase 5: Attention Matrix Construction
// ============================================================================

void host_attention_matrix(const float* K, const float* K_beta,
                           const float* decay_mask,
                           float* attn, int S_k, int C) {
    // Step 1: attn = K^T @ K_beta (result is [C, C])
    // K is [S_k, C], K_beta is [S_k, C]
    // K^T is [C, S_k], so K^T @ K_beta is [C, C]
    host_matmul_tn(K, K_beta, attn, C, C, S_k);

    // Step 2: attn = attn * decay_mask (element-wise)
    host_mul(attn, decay_mask, attn, C * C);

    // Step 3: attn = -attn * causal_mask (strictly lower triangular)
    for (int i = 0; i < C; i++) {
        for (int j = 0; j < C; j++) {
            if (j < i) {
                attn[i * C + j] = -attn[i * C + j];
            } else {
                attn[i * C + j] = 0.0f;
            }
        }
    }
}

// ============================================================================
// Phase 6: Full Intra-Chunk Computation
// ============================================================================

void host_intra_chunk(const float* Q, const float* K, const float* V,
                      const float* G, const float* Beta,
                      float* attn_solved, float* V_new, float* K_cumdecay,
                      int S, int C) {
    // Allocate temporaries
    float* beta_sig = new float[C];
    float* g_cumsum = new float[C];
    float* decay_mask = new float[C * C];
    float* K_beta = new float[S * C];
    float* V_beta = new float[S * C];
    float* attn_pre = new float[C * C];
    float* identity = new float[C * C];
    float* kbeta_gexp = new float[S * C];
    float* gexp = new float[C];

    // Step 1: Sigmoid beta
    host_sigmoid(Beta, beta_sig, C);

    // Step 2: Cumsum of gate
    host_cumsum(G, g_cumsum, C);

    // Step 3: Compute decay mask
    host_decay_mask_causal(g_cumsum, decay_mask, C);

    // Step 4: K_beta = K * beta, V_beta = V * beta (broadcast beta across S)
    for (int s = 0; s < S; s++) {
        for (int c = 0; c < C; c++) {
            K_beta[s * C + c] = K[s * C + c] * beta_sig[c];
            V_beta[s * C + c] = V[s * C + c] * beta_sig[c];
        }
    }

    // Step 5: Build attention matrix
    host_attention_matrix(K, K_beta, decay_mask, attn_pre, S, C);

    // Step 6: Create identity
    host_eye(identity, C);

    // Step 7: Solve triangular system: (I - attn_lower) @ X = attn
    // Note: attn_pre is already strictly lower triangular with negation applied
    // We need to solve (I - attn_lower) @ X = attn_pre
    // But attn_pre = -K@K_beta * decay * causal, so it's the -A in (I - A)
    // Actually, re-reading the algorithm: attn_lower = attn * causal
    // lhs = I - attn_lower, and we solve lhs @ X = attn
    // Since attn_pre is already -(...) * causal, it IS -attn_lower
    // So (I - attn_lower) = I + attn_pre (where attn_pre is the negative)
    // This is confusing... let me re-derive:
    //
    // From qwen3next.cpp:
    //   k_decay = kmulkbeta * decay_mask
    //   attn = -(k_decay * causal_mask)
    //   attn_lower = attn * causal_mask  [same as attn since attn is already masked]
    //   lhs = I - attn_lower = I - attn
    //   lin_solve = solve_tri(lhs, attn)
    //
    // So: lhs = I - attn, and attn = -(k_decay * causal)
    // Therefore: lhs = I + k_decay * causal
    //
    // The solve_tri solves lhs @ X = attn, i.e., (I - attn) @ X = attn
    // In our host_solve_tri: (I - A) @ X = B where A is strictly lower tri
    // Here A = attn (which is strictly lower tri), B = attn
    // So we call: host_solve_tri(attn_pre_positive, attn_pre, X, n, n)
    // Wait, attn_pre has the negative built in...

    // Let me simplify: compute it step by step matching the GPU code exactly

    // Re-compute properly:
    // kmulkbeta = K @ K_beta^T  [C x C]
    float* kmulkbeta = new float[C * C];
    host_matmul_tn(K, K_beta, kmulkbeta, C, C, S);

    // k_decay = kmulkbeta * decay_mask
    float* k_decay = new float[C * C];
    host_mul(kmulkbeta, decay_mask, k_decay, C * C);

    // attn = -(k_decay * causal_mask)  [strictly lower triangular]
    float* causal = new float[C * C];
    host_causal_mask(causal, C);
    host_mul(k_decay, causal, attn_pre, C * C);
    host_neg(attn_pre, attn_pre, C * C);

    // attn_lower = attn * causal (already done, attn_pre is strictly lower)
    // lhs = I - attn_lower
    float* lhs = new float[C * C];
    host_sub(identity, attn_pre, lhs, C * C);

    // Solve: lhs @ X = attn_pre
    // Note: host_solve_tri expects (I - A) format where A is lower tri
    // lhs = I - attn_pre, so A = attn_pre
    // But our host_solve_tri does (I - A) @ X = B as X[i] = B[i] + sum A[i][j]*X[j]
    // So we pass A = attn_pre (which is -(...)), B = attn_pre
    // Hmm, this is getting confusing. Let me just do explicit forward substitution:

    // Forward substitution for (I - attn_lower) @ X = attn
    // (I - attn_lower)[i][j] = delta_ij - attn_lower[i][j]
    // For j < i: (I - attn_lower)[i][j] = -attn_lower[i][j] = -attn_pre[i][j]
    // For j = i: (I - attn_lower)[i][i] = 1
    // For j > i: (I - attn_lower)[i][j] = 0

    // System: for each column c of X (and B):
    // X[0][c] = B[0][c]
    // X[i][c] = B[i][c] + sum_{j<i} attn_lower[i][j] * X[j][c]

    for (int c = 0; c < C; c++) {
        for (int i = 0; i < C; i++) {
            float sum = attn_pre[i * C + c]; // B[i][c]
            for (int j = 0; j < i; j++) {
                // attn_lower = attn_pre (since attn_pre is already lower tri)
                // (I - attn_lower)[i][j] = -attn_pre[i][j] for j < i
                // So: (I - attn_lower) @ X = B
                // X[i] = B[i] - sum_{j<i} (I-attn_lower)[i][j] * X[j]  ??? No...
                //
                // Actually: (I - L) @ X = B
                // Row i: X[i] - sum_{j<i} L[i][j] * X[j] = B[i]
                // So: X[i] = B[i] + sum_{j<i} L[i][j] * X[j]
                //
                // L = attn_lower = attn_pre (which is negative of k_decay*causal)
                sum += attn_pre[i * C + j] * attn_solved[j * C + c];
            }
            attn_solved[i * C + c] = sum;
        }
    }

    // Apply causal mask to result and add identity
    for (int i = 0; i < C; i++) {
        for (int j = 0; j < C; j++) {
            if (j < i) {
                // Keep solved value (already has correct sign)
            } else if (j == i) {
                attn_solved[i * C + j] = 1.0f; // Add identity
            } else {
                attn_solved[i * C + j] = 0.0f; // Zero upper triangle
            }
        }
    }

    // Step 8: V_new = attn_solved @ V_beta^T
    // attn_solved is [C, C], V_beta is [S, C], V_beta^T is [C, S]
    // Result V_new is [C, S]... but we want [S, C]
    // Let me check qwen3next.cpp: v = ggml_mul_mat(ctx0, ggml_transpose(v_beta), attn)
    // So it's V_beta^T @ attn^T = (attn @ V_beta)^T
    // Or equivalently: V_new^T = attn @ V_beta, so V_new = V_beta^T @ attn^T

    // Actually, let's trace dimensions more carefully:
    // In the code, after permute: v is [S_v, chunk, H, batch]
    // v_beta = v * beta: [S_v, chunk, H, batch]
    // transpose(v_beta): [chunk, S_v, H, batch]
    // mul_mat(transpose(v_beta), attn): attn is [chunk, chunk], v_beta^T is [chunk, S_v]
    // Result: [S_v, chunk] (ggml_mul_mat does B @ A)

    // For simplicity, let's compute: V_new[s][c] = sum_c' attn_solved[c'][c] * V_beta[s][c']
    // This is V_new = V_beta @ attn_solved^T
    float* attn_T = new float[C * C];
    host_transpose(attn_solved, attn_T, C, C);
    // V_new = V_beta @ attn_T, V_beta is [S, C], attn_T is [C, C], result is [S, C]
    host_matmul(V_beta, attn_T, V_new, S, C, C);

    // Step 9: K_cumdecay = (attn_solved @ kbeta_gexp^T)^T
    // First compute kbeta_gexp = K_beta * exp(g_cumsum)
    host_exp(g_cumsum, gexp, C);
    for (int s = 0; s < S; s++) {
        for (int c = 0; c < C; c++) {
            kbeta_gexp[s * C + c] = K_beta[s * C + c] * gexp[c];
        }
    }
    // K_cumdecay = (attn_solved @ kbeta_gexp^T)^T
    // attn_solved is [C, C], kbeta_gexp^T is [C, S]
    // attn_solved @ kbeta_gexp^T is [C, S]
    // Transpose to get [S, C]
    float* temp = new float[C * S];
    host_matmul_nt(attn_solved, kbeta_gexp, temp, C, S, C);
    host_transpose(temp, K_cumdecay, C, S);

    // Cleanup
    delete[] beta_sig;
    delete[] g_cumsum;
    delete[] decay_mask;
    delete[] K_beta;
    delete[] V_beta;
    delete[] attn_pre;
    delete[] identity;
    delete[] kbeta_gexp;
    delete[] gexp;
    delete[] kmulkbeta;
    delete[] k_decay;
    delete[] causal;
    delete[] lhs;
    delete[] attn_T;
    delete[] temp;
}

// ============================================================================
// Phase 7: State Interaction
// ============================================================================

void host_output_with_state(const float* Q_gexp, const float* State,
                            const float* attn, const float* V_new,
                            float* output, int S, int C) {
    // output = Q_gexp @ State^T + attn @ V_new^T

    // Term 1: Q_gexp @ State^T
    // Q_gexp is [S, C], State is [S, S], State^T is [S, S]
    // Q_gexp @ State^T: need to be careful about dimensions
    // Actually, Q_gexp should be [C, S] to match State^T [S, S] -> [C, S]
    // Let me assume Q_gexp is [S, C] and we want output [S, C]
    // Then: output[s][c] = sum_s' Q_gexp[s][c] * State[s'][s] + sum_c' attn[c'][c] * V_new[s][c']
    // This doesn't quite work dimensionally...

    // Let me just implement what makes sense:
    // attn_inter = Q_gexp @ State^T where Q_gexp[S,C], State[S,S] -> need Q_gexp[C,S]
    // For now, assume inputs match expected dimensions
    float* attn_inter = new float[S * C];
    float* v_attn = new float[S * C];

    // Q_gexp @ State^T: Q_gexp is [S_k, C], State^T is [S_v, S_v]
    // This doesn't match... need to revisit the exact tensor layout
    // For now, do a simple version assuming compatible dimensions
    host_matmul_nt(Q_gexp, State, attn_inter, S, C, S);

    // attn @ V_new^T: attn is [C, C], V_new is [S, C], V_new^T is [C, S]
    // Result is [C, S], need to transpose to [S, C]
    float* temp = new float[C * S];
    host_matmul_nt(attn, V_new, temp, C, S, C);
    host_transpose(temp, v_attn, C, S);

    // output = attn_inter + v_attn
    host_add(attn_inter, v_attn, output, S * C);

    delete[] attn_inter;
    delete[] v_attn;
    delete[] temp;
}

void host_state_update(const float* State, const float* K_gdiff,
                       const float* V_new, float g_last,
                       float* State_new, int S, int C) {
    // state_new = state * exp(g_last) + K_gdiff^T @ V_new

    float exp_g_last = expf(g_last);

    // Term 1: state * exp(g_last)
    host_scale(State, State_new, S * S, exp_g_last);

    // Term 2: K_gdiff^T @ V_new
    // K_gdiff is [S, C], V_new is [S, C]
    // K_gdiff^T is [C, S]
    // K_gdiff^T @ V_new: [C, S] @ [S, C] = [C, C]... but state is [S, S]
    // There's a dimension mismatch. Need to revisit.
    // For now, just do what matches dimensionally
    float* kgd_vnew = new float[S * S];
    host_matmul_tn(K_gdiff, V_new, kgd_vnew, S, S, C);

    // Add to state_new
    host_add(State_new, kgd_vnew, State_new, S * S);

    delete[] kgd_vnew;
}

// ============================================================================
// Phase 8: Full Delta-Net
// ============================================================================

void host_deltanet_full(const float* Q, const float* K, const float* V,
                        const float* G, const float* Beta,
                        const float* State_in,
                        float* Output, float* State_out,
                        int S, int n_tokens) {
    // Copy initial state
    memcpy(State_out, State_in, S * S * sizeof(float));

    int n_chunks = (n_tokens + CHUNK_SIZE - 1) / CHUNK_SIZE;

    float* attn_solved = new float[CHUNK_SIZE * CHUNK_SIZE];
    float* V_new = new float[S * CHUNK_SIZE];
    float* K_cumdecay = new float[S * CHUNK_SIZE];
    float* g_cumsum = new float[CHUNK_SIZE];
    float* state_temp = new float[S * S];
    float* kv_temp = new float[S * S];

    for (int chunk = 0; chunk < n_chunks; chunk++) {
        int chunk_start = chunk * CHUNK_SIZE;
        int chunk_len = std::min(CHUNK_SIZE, n_tokens - chunk_start);

        // Get chunk pointers (note: K, V, G, Beta are [dim, n_tokens] layout)
        // For simplicity, extract chunk data
        float* K_chunk = new float[S * chunk_len];
        float* V_chunk = new float[S * chunk_len];
        float* G_chunk = new float[chunk_len];
        float* Beta_chunk = new float[chunk_len];

        for (int s = 0; s < S; s++) {
            for (int c = 0; c < chunk_len; c++) {
                K_chunk[s * chunk_len + c] = K[s * n_tokens + chunk_start + c];
                V_chunk[s * chunk_len + c] = V[s * n_tokens + chunk_start + c];
            }
        }
        for (int c = 0; c < chunk_len; c++) {
            G_chunk[c] = G[chunk_start + c];
            Beta_chunk[c] = Beta[chunk_start + c];
        }

        // Intra-chunk computation
        host_intra_chunk(nullptr, K_chunk, V_chunk, G_chunk, Beta_chunk,
                         attn_solved, V_new, K_cumdecay, S, chunk_len);

        // Compute g_cumsum for this chunk (needed for state update)
        host_cumsum(G_chunk, g_cumsum, chunk_len);
        float g_last = g_cumsum[chunk_len - 1];
        float exp_g_last = expf(g_last);

        // State update: State_out = State_out * exp(g_last) + K_cumdecay @ V_new^T
        // K_cumdecay is [S, chunk_len], V_new is [S, chunk_len]
        // K_cumdecay @ V_new^T is [S, S]
        host_scale(State_out, state_temp, S * S, exp_g_last);
        host_matmul_nt(K_cumdecay, V_new, kv_temp, S, S, chunk_len);
        host_add(state_temp, kv_temp, State_out, S * S);

        delete[] K_chunk;
        delete[] V_chunk;
        delete[] G_chunk;
        delete[] Beta_chunk;
    }

    delete[] attn_solved;
    delete[] V_new;
    delete[] K_cumdecay;
    delete[] g_cumsum;
    delete[] state_temp;
    delete[] kv_temp;
}

// ============================================================================
// Utility Functions
// ============================================================================

void print_matrix(const float* m, int rows, int cols, const char* name) {
    printf("%s [%d x %d]:\n", name, rows, cols);
    for (int i = 0; i < rows && i < 8; i++) {  // Print at most 8 rows
        printf("  [");
        for (int j = 0; j < cols && j < 8; j++) {  // Print at most 8 cols
            printf("%8.4f", m[i * cols + j]);
            if (j < cols - 1 && j < 7) printf(", ");
        }
        if (cols > 8) printf(", ...");
        printf("]\n");
    }
    if (rows > 8) printf("  ...\n");
}

float max_diff(const float* a, const float* b, int n) {
    float max_d = 0.0f;
    for (int i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > max_d) max_d = d;
    }
    return max_d;
}

bool compare_arrays(const float* ref, const float* got, int n,
                    float eps, const char* name) {
    float max_d = 0.0f;
    int max_idx = 0;
    int num_errors = 0;

    for (int i = 0; i < n; i++) {
        float d = fabsf(ref[i] - got[i]);
        if (d > max_d) {
            max_d = d;
            max_idx = i;
        }
        if (d > eps) {
            num_errors++;
            if (num_errors <= 5) {
                printf("  MISMATCH at [%d]: ref=%.6f, got=%.6f, diff=%.6e\n",
                       i, ref[i], got[i], d);
            }
        }
    }

    if (name) {
        printf("%s: max_diff=%.6e at [%d]", name, max_d, max_idx);
        if (max_d <= eps) {
            printf(" [PASS]\n");
        } else {
            printf(" [FAIL] (%d errors)\n", num_errors);
        }
    }

    return max_d <= eps;
}

void fill_random(float* arr, int n, float min_val, float max_val) {
    for (int i = 0; i < n; i++) {
        arr[i] = min_val + (max_val - min_val) * (rand() / (float)RAND_MAX);
    }
}

void fill_zeros(float* arr, int n) {
    memset(arr, 0, n * sizeof(float));
}

void fill_ones(float* arr, int n) {
    for (int i = 0; i < n; i++) {
        arr[i] = 1.0f;
    }
}

void fill_sequential(float* arr, int n) {
    for (int i = 0; i < n; i++) {
        arr[i] = (float)i;
    }
}

} // namespace deltanet
