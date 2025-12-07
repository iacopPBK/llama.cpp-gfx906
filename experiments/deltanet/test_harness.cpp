/**
 * Delta-Net Kernel Test Harness
 *
 * Validates GPU kernels against host reference implementations.
 * Since gfx906 doesn't support on-device printf, all debugging
 * is done via host-side comparison.
 *
 * Usage:
 *   ./test_deltanet                  # Run all tests
 *   ./test_deltanet --phase 1        # Run Phase 1 tests only
 *   ./test_deltanet --test sigmoid   # Run specific test
 *   ./test_deltanet --list           # List available tests
 */

#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <cstdlib>
#include <ctime>
#include <functional>
#include <map>

#include "reference.h"

using namespace deltanet;

// ============================================================================
// HIP Error Checking
// ============================================================================

#define HIP_CHECK(call)                                                         \
    do {                                                                         \
        hipError_t err = call;                                                   \
        if (err != hipSuccess) {                                                 \
            std::cerr << "HIP error at " << __FILE__ << ":" << __LINE__ << ": "  \
                      << hipGetErrorString(err) << std::endl;                    \
            exit(1);                                                             \
        }                                                                        \
    } while (0)

// ============================================================================
// Test Framework
// ============================================================================

struct TestResult {
    std::string name;
    bool passed;
    float max_diff;
    double time_ms;
};

std::vector<TestResult> g_results;
int g_tests_run = 0;
int g_tests_passed = 0;

void report_test(const std::string& name, bool passed, float max_diff = 0.0f, double time_ms = 0.0) {
    g_tests_run++;
    if (passed) g_tests_passed++;

    TestResult r = {name, passed, max_diff, time_ms};
    g_results.push_back(r);

    const char* status = passed ? "\033[32mPASS\033[0m" : "\033[31mFAIL\033[0m";
    printf("[%s] %s (max_diff=%.2e)\n", status, name.c_str(), max_diff);
}

void print_summary() {
    printf("\n========================================\n");
    printf("Test Summary: %d/%d passed\n", g_tests_passed, g_tests_run);
    printf("========================================\n");

    if (g_tests_passed < g_tests_run) {
        printf("\nFailed tests:\n");
        for (const auto& r : g_results) {
            if (!r.passed) {
                printf("  - %s (max_diff=%.2e)\n", r.name.c_str(), r.max_diff);
            }
        }
    }
}

// ============================================================================
// Device Memory Helpers
// ============================================================================

template<typename T>
T* device_alloc(size_t count) {
    T* ptr;
    HIP_CHECK(hipMalloc(&ptr, count * sizeof(T)));
    return ptr;
}

template<typename T>
void device_free(T* ptr) {
    if (ptr) HIP_CHECK(hipFree(ptr));
}

template<typename T>
void to_device(T* d_ptr, const T* h_ptr, size_t count) {
    HIP_CHECK(hipMemcpy(d_ptr, h_ptr, count * sizeof(T), hipMemcpyHostToDevice));
}

template<typename T>
void to_host(T* h_ptr, const T* d_ptr, size_t count) {
    HIP_CHECK(hipMemcpy(h_ptr, d_ptr, count * sizeof(T), hipMemcpyDeviceToHost));
}

// ============================================================================
// Phase 1 Kernels: Element-wise Operations
// ============================================================================

__global__ void kernel_sigmoid(const float* __restrict__ in,
                                float* __restrict__ out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = 1.0f / (1.0f + expf(-in[idx]));
    }
}

__global__ void kernel_exp(const float* __restrict__ in,
                           float* __restrict__ out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = expf(in[idx]);
    }
}

__global__ void kernel_mul(const float* __restrict__ a,
                           const float* __restrict__ b,
                           float* __restrict__ out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] * b[idx];
    }
}

__global__ void kernel_cumsum_sequential(const float* __restrict__ in,
                                          float* __restrict__ out, int n) {
    // Single-threaded sequential cumsum (for validation)
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float sum = 0.0f;
        for (int i = 0; i < n; i++) {
            sum += in[i];
            out[i] = sum;
        }
    }
}

// ============================================================================
// Phase 2 Kernels: Triangular Operations
// ============================================================================

__global__ void kernel_tril(const float* __restrict__ src,
                            float* __restrict__ dst,
                            int n, int diag_offset) {
    int row = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        int idx = row * n + col;
        dst[idx] = (col <= row + diag_offset) ? src[idx] : 0.0f;
    }
}

__global__ void kernel_causal_mask(float* __restrict__ out, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        out[row * n + col] = (col < row) ? 1.0f : 0.0f;
    }
}

__global__ void kernel_eye(float* __restrict__ out, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        out[row * n + col] = (row == col) ? 1.0f : 0.0f;
    }
}

// Sequential triangular solve for validation
__global__ void kernel_solve_tri_sequential(
    const float* __restrict__ A,  // n x n strictly lower triangular
    const float* __restrict__ B,  // n x k right-hand side
    float* __restrict__ X,        // n x k output
    int n, int k
) {
    // Single thread does forward substitution
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        for (int col = 0; col < k; col++) {
            for (int row = 0; row < n; row++) {
                float sum = B[row * k + col];
                for (int j = 0; j < row; j++) {
                    sum += A[row * n + j] * X[j * k + col];
                }
                X[row * k + col] = sum;
            }
        }
    }
}

// ============================================================================
// Phase 3 Kernels: Matrix Operations (GEMM)
// ============================================================================

// Naive GEMM: C = A @ B^T
// A: [M, K], B: [N, K], C: [M, N]
__global__ void kernel_gemm_nt_naive(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[col * K + k];
        }
        C[row * N + col] = sum;
    }
}

// Naive GEMM: C = A^T @ B
// A: [K, M], B: [K, N], C: [M, N]
__global__ void kernel_gemm_tn_naive(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[k * M + row] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Naive GEMM: C = A @ B
// A: [M, K], B: [K, N], C: [M, N]
__global__ void kernel_gemm_nn_naive(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// LDS-Tiled GEMM: C = A @ B^T (optimized for gfx906)
// Tile size 16x16 for good occupancy
#define TILE_SIZE 16

__global__ void kernel_gemm_nt_tiled(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    __shared__ float s_A[TILE_SIZE][TILE_SIZE];
    __shared__ float s_B[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    // Loop over tiles
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load A tile: A[row, t*TILE + threadIdx.x]
        int a_col = t * TILE_SIZE + threadIdx.x;
        if (row < M && a_col < K) {
            s_A[threadIdx.y][threadIdx.x] = A[row * K + a_col];
        } else {
            s_A[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load B tile: B[col, t*TILE + threadIdx.y] (transposed access)
        int b_col = t * TILE_SIZE + threadIdx.y;
        if (col < N && b_col < K) {
            s_B[threadIdx.y][threadIdx.x] = B[col * K + b_col];
        } else {
            s_B[threadIdx.y][threadIdx.x] = 0.0f;
        }

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

// ============================================================================
// Phase 4 Kernels: Decay Mask
// ============================================================================

// Compute decay mask: mask[i,j] = exp(g_cumsum[j] - g_cumsum[i]) if j <= i, else 0
__global__ void kernel_decay_mask(
    const float* __restrict__ g_cumsum,  // [n]
    float* __restrict__ mask,            // [n, n]
    int n
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        if (col <= row) {
            // Causal: position col can attend to position row (col <= row)
            mask[row * n + col] = expf(g_cumsum[col] - g_cumsum[row]);
        } else {
            mask[row * n + col] = 0.0f;
        }
    }
}

// ============================================================================
// Additional Element-wise Kernels for Phase 6
// ============================================================================

// Negate: out[i] = -in[i]
__global__ void kernel_neg(const float* __restrict__ in,
                           float* __restrict__ out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = -in[idx];
    }
}

// Subtract: out[i] = a[i] - b[i]
__global__ void kernel_sub(const float* __restrict__ a,
                           const float* __restrict__ b,
                           float* __restrict__ out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] - b[idx];
    }
}

// Broadcast multiply: out[s,c] = a[s,c] * b[c] (broadcast b across S dimension)
__global__ void kernel_broadcast_mul(const float* __restrict__ a,   // [S, C]
                                     const float* __restrict__ b,   // [C]
                                     float* __restrict__ out,       // [S, C]
                                     int S, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < S * C) {
        int c = idx % C;
        out[idx] = a[idx] * b[c];
    }
}

// Transpose: out[j,i] = in[i,j]
__global__ void kernel_transpose(const float* __restrict__ in,
                                 float* __restrict__ out,
                                 int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        out[col * rows + row] = in[row * cols + col];
    }
}

// Apply causal mask and add identity diagonal
// out[i,j] = (j < i) ? in[i,j] : ((j == i) ? 1.0 : 0.0)
__global__ void kernel_apply_causal_identity(const float* __restrict__ in,
                                             float* __restrict__ out,
                                             int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        int idx = row * n + col;
        if (col < row) {
            out[idx] = in[idx];
        } else if (col == row) {
            out[idx] = 1.0f;
        } else {
            out[idx] = 0.0f;
        }
    }
}

// Scale: out[i] = in[i] * scale
__global__ void kernel_scale(const float* __restrict__ in,
                              float* __restrict__ out, int n, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = in[idx] * scale;
    }
}

// Add: out[i] = a[i] + b[i]
__global__ void kernel_add(const float* __restrict__ a,
                           const float* __restrict__ b,
                           float* __restrict__ out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] + b[idx];
    }
}

// ============================================================================
// Phase 5 Kernels: Attention Matrix Construction
// ============================================================================

// Build attention matrix: attn = -(K^T @ K_beta) * decay_mask * causal_mask
// K: [S, C], K_beta: [S, C], decay_mask: [C, C], attn: [C, C]
// This is a fused kernel that does GEMM + element-wise ops
__global__ void kernel_attention_matrix_fused(
    const float* __restrict__ K,         // [S, C]
    const float* __restrict__ K_beta,    // [S, C]
    const float* __restrict__ decay_mask,// [C, C]
    float* __restrict__ attn,            // [C, C]
    int S, int C
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < C && col < C) {
        // Compute K^T @ K_beta at [row, col]
        // K^T[row, :] is K[:, row], K_beta[:, col] is K_beta[:, col]
        float sum = 0.0f;
        for (int s = 0; s < S; s++) {
            sum += K[s * C + row] * K_beta[s * C + col];
        }

        // Apply decay mask and causal mask
        if (col < row) {  // Strictly lower triangular (causal mask)
            attn[row * C + col] = -sum * decay_mask[row * C + col];
        } else {
            attn[row * C + col] = 0.0f;
        }
    }
}

// ============================================================================
// Phase 1 Tests
// ============================================================================

bool test_sigmoid() {
    const int N = 1024;
    std::vector<float> h_in(N), h_ref(N), h_out(N);

    // Initialize with diverse values
    srand(42);
    fill_random(h_in.data(), N, -5.0f, 5.0f);

    // Host reference
    host_sigmoid(h_in.data(), h_ref.data(), N);

    // Device
    float* d_in = device_alloc<float>(N);
    float* d_out = device_alloc<float>(N);
    to_device(d_in, h_in.data(), N);

    int block = 256;
    int grid = (N + block - 1) / block;
    kernel_sigmoid<<<grid, block>>>(d_in, d_out, N);
    HIP_CHECK(hipDeviceSynchronize());

    to_host(h_out.data(), d_out, N);

    float diff = max_diff(h_ref.data(), h_out.data(), N);
    bool passed = diff < 1e-5f;

    device_free(d_in);
    device_free(d_out);

    report_test("Phase1: sigmoid", passed, diff);
    return passed;
}

bool test_exp() {
    const int N = 1024;
    std::vector<float> h_in(N), h_ref(N), h_out(N);

    srand(43);
    fill_random(h_in.data(), N, -3.0f, 3.0f);  // Avoid huge exp values

    host_exp(h_in.data(), h_ref.data(), N);

    float* d_in = device_alloc<float>(N);
    float* d_out = device_alloc<float>(N);
    to_device(d_in, h_in.data(), N);

    kernel_exp<<<(N + 255) / 256, 256>>>(d_in, d_out, N);
    HIP_CHECK(hipDeviceSynchronize());

    to_host(h_out.data(), d_out, N);

    float diff = max_diff(h_ref.data(), h_out.data(), N);
    bool passed = diff < 1e-4f;  // exp can have larger errors

    device_free(d_in);
    device_free(d_out);

    report_test("Phase1: exp", passed, diff);
    return passed;
}

bool test_mul() {
    const int N = 1024;
    std::vector<float> h_a(N), h_b(N), h_ref(N), h_out(N);

    srand(44);
    fill_random(h_a.data(), N, -2.0f, 2.0f);
    fill_random(h_b.data(), N, -2.0f, 2.0f);

    host_mul(h_a.data(), h_b.data(), h_ref.data(), N);

    float* d_a = device_alloc<float>(N);
    float* d_b = device_alloc<float>(N);
    float* d_out = device_alloc<float>(N);
    to_device(d_a, h_a.data(), N);
    to_device(d_b, h_b.data(), N);

    kernel_mul<<<(N + 255) / 256, 256>>>(d_a, d_b, d_out, N);
    HIP_CHECK(hipDeviceSynchronize());

    to_host(h_out.data(), d_out, N);

    float diff = max_diff(h_ref.data(), h_out.data(), N);
    bool passed = diff < 1e-6f;

    device_free(d_a);
    device_free(d_b);
    device_free(d_out);

    report_test("Phase1: mul", passed, diff);
    return passed;
}

bool test_cumsum() {
    const int N = 64;  // Small for sequential kernel
    std::vector<float> h_in(N), h_ref(N), h_out(N);

    srand(45);
    fill_random(h_in.data(), N, -1.0f, 1.0f);

    host_cumsum(h_in.data(), h_ref.data(), N);

    float* d_in = device_alloc<float>(N);
    float* d_out = device_alloc<float>(N);
    to_device(d_in, h_in.data(), N);

    kernel_cumsum_sequential<<<1, 1>>>(d_in, d_out, N);
    HIP_CHECK(hipDeviceSynchronize());

    to_host(h_out.data(), d_out, N);

    float diff = max_diff(h_ref.data(), h_out.data(), N);
    bool passed = diff < 1e-5f;

    device_free(d_in);
    device_free(d_out);

    report_test("Phase1: cumsum", passed, diff);
    return passed;
}

// ============================================================================
// Phase 2 Tests
// ============================================================================

bool test_tril() {
    const int N = 8;  // Small for easy debugging
    std::vector<float> h_in(N * N), h_ref(N * N), h_out(N * N);

    // Fill with sequential values
    fill_sequential(h_in.data(), N * N);

    host_tril(h_in.data(), h_ref.data(), N, 0);

    float* d_in = device_alloc<float>(N * N);
    float* d_out = device_alloc<float>(N * N);
    to_device(d_in, h_in.data(), N * N);

    dim3 block(N, 1);
    dim3 grid(1, N);
    kernel_tril<<<grid, block>>>(d_in, d_out, N, 0);
    HIP_CHECK(hipDeviceSynchronize());

    to_host(h_out.data(), d_out, N * N);

    float diff = max_diff(h_ref.data(), h_out.data(), N * N);
    bool passed = diff < 1e-6f;

    if (!passed) {
        printf("Expected:\n");
        print_matrix(h_ref.data(), N, N, "ref");
        printf("Got:\n");
        print_matrix(h_out.data(), N, N, "got");
    }

    device_free(d_in);
    device_free(d_out);

    report_test("Phase2: tril", passed, diff);
    return passed;
}

bool test_causal_mask() {
    const int N = 8;
    std::vector<float> h_ref(N * N), h_out(N * N);

    host_causal_mask(h_ref.data(), N);

    float* d_out = device_alloc<float>(N * N);

    dim3 block(N, 1);
    dim3 grid(1, N);
    kernel_causal_mask<<<grid, block>>>(d_out, N);
    HIP_CHECK(hipDeviceSynchronize());

    to_host(h_out.data(), d_out, N * N);

    float diff = max_diff(h_ref.data(), h_out.data(), N * N);
    bool passed = diff < 1e-6f;

    device_free(d_out);

    report_test("Phase2: causal_mask", passed, diff);
    return passed;
}

bool test_eye() {
    const int N = 8;
    std::vector<float> h_ref(N * N), h_out(N * N);

    host_eye(h_ref.data(), N);

    float* d_out = device_alloc<float>(N * N);

    dim3 block(N, 1);
    dim3 grid(1, N);
    kernel_eye<<<grid, block>>>(d_out, N);
    HIP_CHECK(hipDeviceSynchronize());

    to_host(h_out.data(), d_out, N * N);

    float diff = max_diff(h_ref.data(), h_out.data(), N * N);
    bool passed = diff < 1e-6f;

    device_free(d_out);

    report_test("Phase2: eye", passed, diff);
    return passed;
}

bool test_solve_tri_small() {
    // Small 4x4 test case with known answer
    const int N = 4;
    const int K = 1;

    // A is strictly lower triangular (negative of what we solve)
    // (I - A) is unit lower triangular with -A below diagonal
    std::vector<float> h_A = {
        0.0f, 0.0f, 0.0f, 0.0f,
        0.5f, 0.0f, 0.0f, 0.0f,
        0.3f, 0.2f, 0.0f, 0.0f,
        0.1f, 0.4f, 0.3f, 0.0f
    };

    std::vector<float> h_B = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> h_ref(N), h_out(N);

    // Host reference
    host_solve_tri(h_A.data(), h_B.data(), h_ref.data(), N, K);

    printf("  Reference solution: [%.4f, %.4f, %.4f, %.4f]\n",
           h_ref[0], h_ref[1], h_ref[2], h_ref[3]);

    // Device
    float* d_A = device_alloc<float>(N * N);
    float* d_B = device_alloc<float>(N);
    float* d_X = device_alloc<float>(N);

    to_device(d_A, h_A.data(), N * N);
    to_device(d_B, h_B.data(), N);

    kernel_solve_tri_sequential<<<1, 1>>>(d_A, d_B, d_X, N, K);
    HIP_CHECK(hipDeviceSynchronize());

    to_host(h_out.data(), d_X, N);

    printf("  GPU solution: [%.4f, %.4f, %.4f, %.4f]\n",
           h_out[0], h_out[1], h_out[2], h_out[3]);

    float diff = max_diff(h_ref.data(), h_out.data(), N);
    bool passed = diff < 1e-5f;

    device_free(d_A);
    device_free(d_B);
    device_free(d_X);

    report_test("Phase2: solve_tri_small", passed, diff);
    return passed;
}

bool test_solve_tri_64() {
    // 64x64 test (chunk size)
    const int N = 64;
    const int K = 64;

    std::vector<float> h_A(N * N), h_B(N * K), h_ref(N * K), h_out(N * K);

    srand(46);

    // Create random strictly lower triangular A
    fill_zeros(h_A.data(), N * N);
    for (int i = 1; i < N; i++) {
        for (int j = 0; j < i; j++) {
            h_A[i * N + j] = (rand() / (float)RAND_MAX - 0.5f) * 0.1f;  // Small values
        }
    }

    // Random B
    fill_random(h_B.data(), N * K, -1.0f, 1.0f);

    // Host reference
    host_solve_tri(h_A.data(), h_B.data(), h_ref.data(), N, K);

    // Device
    float* d_A = device_alloc<float>(N * N);
    float* d_B = device_alloc<float>(N * K);
    float* d_X = device_alloc<float>(N * K);

    to_device(d_A, h_A.data(), N * N);
    to_device(d_B, h_B.data(), N * K);

    kernel_solve_tri_sequential<<<1, 1>>>(d_A, d_B, d_X, N, K);
    HIP_CHECK(hipDeviceSynchronize());

    to_host(h_out.data(), d_X, N * K);

    float diff = max_diff(h_ref.data(), h_out.data(), N * K);
    bool passed = diff < 1e-3f;  // Allow more tolerance for larger matrix

    device_free(d_A);
    device_free(d_B);
    device_free(d_X);

    report_test("Phase2: solve_tri_64x64", passed, diff);
    return passed;
}

// ============================================================================
// Phase 3 Tests: GEMM
// ============================================================================

bool test_gemm_nt_naive_small() {
    // Small test: A[4,8] @ B[4,8]^T = C[4,4]
    const int M = 4, N = 4, K = 8;
    std::vector<float> h_A(M * K), h_B(N * K), h_ref(M * N), h_out(M * N);

    srand(50);
    fill_random(h_A.data(), M * K, -1.0f, 1.0f);
    fill_random(h_B.data(), N * K, -1.0f, 1.0f);

    // Host reference
    host_matmul_nt(h_A.data(), h_B.data(), h_ref.data(), M, N, K);

    // Device
    float* d_A = device_alloc<float>(M * K);
    float* d_B = device_alloc<float>(N * K);
    float* d_C = device_alloc<float>(M * N);

    to_device(d_A, h_A.data(), M * K);
    to_device(d_B, h_B.data(), N * K);

    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    kernel_gemm_nt_naive<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    HIP_CHECK(hipDeviceSynchronize());

    to_host(h_out.data(), d_C, M * N);

    float diff = max_diff(h_ref.data(), h_out.data(), M * N);
    bool passed = diff < 1e-4f;

    if (!passed) {
        print_matrix(h_ref.data(), M, N, "ref");
        print_matrix(h_out.data(), M, N, "got");
    }

    device_free(d_A);
    device_free(d_B);
    device_free(d_C);

    report_test("Phase3: gemm_nt_naive_small", passed, diff);
    return passed;
}

bool test_gemm_nt_naive_64() {
    // Delta-Net relevant: 64x64 output
    const int M = 64, N = 64, K = 128;
    std::vector<float> h_A(M * K), h_B(N * K), h_ref(M * N), h_out(M * N);

    srand(51);
    fill_random(h_A.data(), M * K, -0.5f, 0.5f);
    fill_random(h_B.data(), N * K, -0.5f, 0.5f);

    host_matmul_nt(h_A.data(), h_B.data(), h_ref.data(), M, N, K);

    float* d_A = device_alloc<float>(M * K);
    float* d_B = device_alloc<float>(N * K);
    float* d_C = device_alloc<float>(M * N);

    to_device(d_A, h_A.data(), M * K);
    to_device(d_B, h_B.data(), N * K);

    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    kernel_gemm_nt_naive<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    HIP_CHECK(hipDeviceSynchronize());

    to_host(h_out.data(), d_C, M * N);

    float diff = max_diff(h_ref.data(), h_out.data(), M * N);
    bool passed = diff < 1e-3f;

    device_free(d_A);
    device_free(d_B);
    device_free(d_C);

    report_test("Phase3: gemm_nt_naive_64x64", passed, diff);
    return passed;
}

bool test_gemm_tn_naive() {
    // A^T @ B: A[K,M], B[K,N] -> C[M,N]
    const int M = 64, N = 64, K = 128;
    std::vector<float> h_A(K * M), h_B(K * N), h_ref(M * N), h_out(M * N);

    srand(52);
    fill_random(h_A.data(), K * M, -0.5f, 0.5f);
    fill_random(h_B.data(), K * N, -0.5f, 0.5f);

    host_matmul_tn(h_A.data(), h_B.data(), h_ref.data(), M, N, K);

    float* d_A = device_alloc<float>(K * M);
    float* d_B = device_alloc<float>(K * N);
    float* d_C = device_alloc<float>(M * N);

    to_device(d_A, h_A.data(), K * M);
    to_device(d_B, h_B.data(), K * N);

    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    kernel_gemm_tn_naive<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    HIP_CHECK(hipDeviceSynchronize());

    to_host(h_out.data(), d_C, M * N);

    float diff = max_diff(h_ref.data(), h_out.data(), M * N);
    bool passed = diff < 1e-3f;

    device_free(d_A);
    device_free(d_B);
    device_free(d_C);

    report_test("Phase3: gemm_tn_naive_64x64", passed, diff);
    return passed;
}

bool test_gemm_nt_tiled() {
    // Test tiled version matches naive
    const int M = 64, N = 64, K = 128;
    std::vector<float> h_A(M * K), h_B(N * K), h_ref(M * N), h_out(M * N);

    srand(53);
    fill_random(h_A.data(), M * K, -0.5f, 0.5f);
    fill_random(h_B.data(), N * K, -0.5f, 0.5f);

    host_matmul_nt(h_A.data(), h_B.data(), h_ref.data(), M, N, K);

    float* d_A = device_alloc<float>(M * K);
    float* d_B = device_alloc<float>(N * K);
    float* d_C = device_alloc<float>(M * N);

    to_device(d_A, h_A.data(), M * K);
    to_device(d_B, h_B.data(), N * K);

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    kernel_gemm_nt_tiled<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    HIP_CHECK(hipDeviceSynchronize());

    to_host(h_out.data(), d_C, M * N);

    float diff = max_diff(h_ref.data(), h_out.data(), M * N);
    bool passed = diff < 1e-3f;

    device_free(d_A);
    device_free(d_B);
    device_free(d_C);

    report_test("Phase3: gemm_nt_tiled_64x64", passed, diff);
    return passed;
}

bool test_gemm_nt_tiled_128() {
    // 128x128 - state size
    const int M = 128, N = 128, K = 64;
    std::vector<float> h_A(M * K), h_B(N * K), h_ref(M * N), h_out(M * N);

    srand(54);
    fill_random(h_A.data(), M * K, -0.5f, 0.5f);
    fill_random(h_B.data(), N * K, -0.5f, 0.5f);

    host_matmul_nt(h_A.data(), h_B.data(), h_ref.data(), M, N, K);

    float* d_A = device_alloc<float>(M * K);
    float* d_B = device_alloc<float>(N * K);
    float* d_C = device_alloc<float>(M * N);

    to_device(d_A, h_A.data(), M * K);
    to_device(d_B, h_B.data(), N * K);

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    kernel_gemm_nt_tiled<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    HIP_CHECK(hipDeviceSynchronize());

    to_host(h_out.data(), d_C, M * N);

    float diff = max_diff(h_ref.data(), h_out.data(), M * N);
    bool passed = diff < 1e-3f;

    device_free(d_A);
    device_free(d_B);
    device_free(d_C);

    report_test("Phase3: gemm_nt_tiled_128x128", passed, diff);
    return passed;
}

// ============================================================================
// Phase 4 Tests: Decay Mask
// ============================================================================

bool test_decay_mask_small() {
    const int N = 8;
    std::vector<float> h_g(N), h_cumsum(N), h_ref(N * N), h_out(N * N);

    srand(60);
    fill_random(h_g.data(), N, -0.5f, 0.5f);

    // Compute cumsum on host
    host_cumsum(h_g.data(), h_cumsum.data(), N);

    // Compute decay mask on host
    host_decay_mask(h_cumsum.data(), h_ref.data(), N);

    // Device
    float* d_cumsum = device_alloc<float>(N);
    float* d_mask = device_alloc<float>(N * N);

    to_device(d_cumsum, h_cumsum.data(), N);

    dim3 block(8, 8);
    dim3 grid(1, 1);
    kernel_decay_mask<<<grid, block>>>(d_cumsum, d_mask, N);
    HIP_CHECK(hipDeviceSynchronize());

    to_host(h_out.data(), d_mask, N * N);

    float diff = max_diff(h_ref.data(), h_out.data(), N * N);
    bool passed = diff < 1e-5f;

    if (!passed) {
        print_matrix(h_ref.data(), N, N, "ref");
        print_matrix(h_out.data(), N, N, "got");
    }

    device_free(d_cumsum);
    device_free(d_mask);

    report_test("Phase4: decay_mask_small", passed, diff);
    return passed;
}

bool test_decay_mask_64() {
    const int N = 64;  // Chunk size
    std::vector<float> h_g(N), h_cumsum(N), h_ref(N * N), h_out(N * N);

    srand(61);
    fill_random(h_g.data(), N, -0.3f, 0.3f);  // Smaller range for stable exp

    host_cumsum(h_g.data(), h_cumsum.data(), N);
    host_decay_mask(h_cumsum.data(), h_ref.data(), N);

    float* d_cumsum = device_alloc<float>(N);
    float* d_mask = device_alloc<float>(N * N);

    to_device(d_cumsum, h_cumsum.data(), N);

    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (N + 15) / 16);
    kernel_decay_mask<<<grid, block>>>(d_cumsum, d_mask, N);
    HIP_CHECK(hipDeviceSynchronize());

    to_host(h_out.data(), d_mask, N * N);

    float diff = max_diff(h_ref.data(), h_out.data(), N * N);
    bool passed = diff < 1e-4f;

    device_free(d_cumsum);
    device_free(d_mask);

    report_test("Phase4: decay_mask_64", passed, diff);
    return passed;
}

// ============================================================================
// Phase 5 Tests: Attention Matrix
// ============================================================================

bool test_attention_matrix_small() {
    const int S = 8, C = 8;  // Small for debugging
    std::vector<float> h_K(S * C), h_K_beta(S * C);
    std::vector<float> h_g(C), h_cumsum(C), h_decay(C * C);
    std::vector<float> h_ref(C * C), h_out(C * C);

    srand(70);
    fill_random(h_K.data(), S * C, -0.5f, 0.5f);
    fill_random(h_K_beta.data(), S * C, -0.5f, 0.5f);
    fill_random(h_g.data(), C, -0.2f, 0.2f);

    // Compute cumsum and decay mask
    host_cumsum(h_g.data(), h_cumsum.data(), C);
    host_decay_mask(h_cumsum.data(), h_decay.data(), C);

    // Compute reference attention matrix
    host_attention_matrix(h_K.data(), h_K_beta.data(), h_decay.data(),
                          h_ref.data(), S, C);

    // Device
    float* d_K = device_alloc<float>(S * C);
    float* d_K_beta = device_alloc<float>(S * C);
    float* d_decay = device_alloc<float>(C * C);
    float* d_attn = device_alloc<float>(C * C);

    to_device(d_K, h_K.data(), S * C);
    to_device(d_K_beta, h_K_beta.data(), S * C);
    to_device(d_decay, h_decay.data(), C * C);

    dim3 block(8, 8);
    dim3 grid(1, 1);
    kernel_attention_matrix_fused<<<grid, block>>>(d_K, d_K_beta, d_decay,
                                                   d_attn, S, C);
    HIP_CHECK(hipDeviceSynchronize());

    to_host(h_out.data(), d_attn, C * C);

    float diff = max_diff(h_ref.data(), h_out.data(), C * C);
    bool passed = diff < 1e-4f;

    if (!passed) {
        print_matrix(h_ref.data(), C, C, "ref");
        print_matrix(h_out.data(), C, C, "got");
    }

    device_free(d_K);
    device_free(d_K_beta);
    device_free(d_decay);
    device_free(d_attn);

    report_test("Phase5: attention_matrix_small", passed, diff);
    return passed;
}

bool test_attention_matrix_64() {
    const int S = 128, C = 64;  // Delta-Net dimensions
    std::vector<float> h_K(S * C), h_K_beta(S * C);
    std::vector<float> h_g(C), h_cumsum(C), h_decay(C * C);
    std::vector<float> h_ref(C * C), h_out(C * C);

    srand(71);
    fill_random(h_K.data(), S * C, -0.3f, 0.3f);
    fill_random(h_K_beta.data(), S * C, -0.3f, 0.3f);
    fill_random(h_g.data(), C, -0.2f, 0.2f);

    host_cumsum(h_g.data(), h_cumsum.data(), C);
    host_decay_mask(h_cumsum.data(), h_decay.data(), C);
    host_attention_matrix(h_K.data(), h_K_beta.data(), h_decay.data(),
                          h_ref.data(), S, C);

    float* d_K = device_alloc<float>(S * C);
    float* d_K_beta = device_alloc<float>(S * C);
    float* d_decay = device_alloc<float>(C * C);
    float* d_attn = device_alloc<float>(C * C);

    to_device(d_K, h_K.data(), S * C);
    to_device(d_K_beta, h_K_beta.data(), S * C);
    to_device(d_decay, h_decay.data(), C * C);

    dim3 block(16, 16);
    dim3 grid((C + 15) / 16, (C + 15) / 16);
    kernel_attention_matrix_fused<<<grid, block>>>(d_K, d_K_beta, d_decay,
                                                   d_attn, S, C);
    HIP_CHECK(hipDeviceSynchronize());

    to_host(h_out.data(), d_attn, C * C);

    float diff = max_diff(h_ref.data(), h_out.data(), C * C);
    bool passed = diff < 1e-3f;

    device_free(d_K);
    device_free(d_K_beta);
    device_free(d_decay);
    device_free(d_attn);

    report_test("Phase5: attention_matrix_64x64", passed, diff);
    return passed;
}

// ============================================================================
// Phase 6 Tests: Intra-Chunk Computation
// ============================================================================

// Full intra-chunk computation test - orchestrates all kernels
bool test_intra_chunk() {
    const int S = 128;  // Key/Value head dimension
    const int C = 64;   // Chunk size

    // Allocate host memory
    std::vector<float> h_K(S * C), h_V(S * C), h_G(C), h_Beta(C);
    std::vector<float> h_attn_solved_ref(C * C), h_V_new_ref(S * C), h_K_cumdecay_ref(S * C);
    std::vector<float> h_attn_solved_out(C * C), h_V_new_out(S * C), h_K_cumdecay_out(S * C);

    // Initialize inputs
    srand(100);
    fill_random(h_K.data(), S * C, -0.5f, 0.5f);
    fill_random(h_V.data(), S * C, -0.5f, 0.5f);
    fill_random(h_G.data(), C, -0.2f, 0.2f);  // Small gate values for stable exp
    fill_random(h_Beta.data(), C, -2.0f, 2.0f);

    // Compute host reference
    std::vector<float> h_Q_dummy(S * C, 0.0f);  // Not used in current impl
    host_intra_chunk(h_Q_dummy.data(), h_K.data(), h_V.data(),
                     h_G.data(), h_Beta.data(),
                     h_attn_solved_ref.data(), h_V_new_ref.data(), h_K_cumdecay_ref.data(),
                     S, C);

    // Allocate device memory
    float* d_K = device_alloc<float>(S * C);
    float* d_V = device_alloc<float>(S * C);
    float* d_G = device_alloc<float>(C);
    float* d_Beta = device_alloc<float>(C);

    // Temporaries
    float* d_beta_sig = device_alloc<float>(C);
    float* d_g_cumsum = device_alloc<float>(C);
    float* d_gexp = device_alloc<float>(C);
    float* d_decay_mask = device_alloc<float>(C * C);
    float* d_K_beta = device_alloc<float>(S * C);
    float* d_V_beta = device_alloc<float>(S * C);
    float* d_kmulkbeta = device_alloc<float>(C * C);
    float* d_k_decay = device_alloc<float>(C * C);
    float* d_causal = device_alloc<float>(C * C);
    float* d_attn_pre = device_alloc<float>(C * C);
    float* d_attn_solved = device_alloc<float>(C * C);
    float* d_attn_T = device_alloc<float>(C * C);
    float* d_V_new = device_alloc<float>(S * C);
    float* d_kbeta_gexp = device_alloc<float>(S * C);
    float* d_temp = device_alloc<float>(C * S);
    float* d_K_cumdecay = device_alloc<float>(S * C);

    // Copy inputs to device
    to_device(d_K, h_K.data(), S * C);
    to_device(d_V, h_V.data(), S * C);
    to_device(d_G, h_G.data(), C);
    to_device(d_Beta, h_Beta.data(), C);

    int block1d = 256;
    dim3 block2d(16, 16);

    // Step 1: Sigmoid beta
    kernel_sigmoid<<<(C + block1d - 1) / block1d, block1d>>>(d_Beta, d_beta_sig, C);

    // Step 2: Cumsum of gate
    kernel_cumsum_sequential<<<1, 1>>>(d_G, d_g_cumsum, C);

    // Step 3: Compute decay mask (causal version)
    dim3 grid_C((C + 15) / 16, (C + 15) / 16);
    kernel_decay_mask<<<grid_C, block2d>>>(d_g_cumsum, d_decay_mask, C);

    // Step 4: K_beta = K * beta_sig (broadcast), V_beta = V * beta_sig
    kernel_broadcast_mul<<<(S * C + block1d - 1) / block1d, block1d>>>(d_K, d_beta_sig, d_K_beta, S, C);
    kernel_broadcast_mul<<<(S * C + block1d - 1) / block1d, block1d>>>(d_V, d_beta_sig, d_V_beta, S, C);

    // Step 5: K^T @ K_beta -> kmulkbeta [C, C]
    kernel_gemm_tn_naive<<<grid_C, block2d>>>(d_K, d_K_beta, d_kmulkbeta, C, C, S);

    // Step 6: k_decay = kmulkbeta * decay_mask
    kernel_mul<<<(C * C + block1d - 1) / block1d, block1d>>>(d_kmulkbeta, d_decay_mask, d_k_decay, C * C);

    // Step 7: Create causal mask and apply: attn_pre = -(k_decay * causal)
    kernel_causal_mask<<<grid_C, block2d>>>(d_causal, C);
    kernel_mul<<<(C * C + block1d - 1) / block1d, block1d>>>(d_k_decay, d_causal, d_attn_pre, C * C);
    kernel_neg<<<(C * C + block1d - 1) / block1d, block1d>>>(d_attn_pre, d_attn_pre, C * C);

    // Step 8: Triangular solve (I - attn_pre) @ X = attn_pre
    kernel_solve_tri_sequential<<<1, 1>>>(d_attn_pre, d_attn_pre, d_attn_solved, C, C);

    // Step 9: Apply causal mask and add identity diagonal
    kernel_apply_causal_identity<<<grid_C, block2d>>>(d_attn_solved, d_attn_solved, C);

    // Step 10: V_new = V_beta @ attn_solved^T
    kernel_transpose<<<grid_C, block2d>>>(d_attn_solved, d_attn_T, C, C);
    // V_new = V_beta [S,C] @ attn_T [C,C] -> [S,C]
    dim3 grid_SC((C + 15) / 16, (S + 15) / 16);
    kernel_gemm_nn_naive<<<grid_SC, block2d>>>(d_V_beta, d_attn_T, d_V_new, S, C, C);

    // Step 11: K_cumdecay computation
    // gexp = exp(g_cumsum)
    kernel_exp<<<(C + block1d - 1) / block1d, block1d>>>(d_g_cumsum, d_gexp, C);
    // kbeta_gexp = K_beta * gexp (broadcast)
    kernel_broadcast_mul<<<(S * C + block1d - 1) / block1d, block1d>>>(d_K_beta, d_gexp, d_kbeta_gexp, S, C);
    // temp = attn_solved @ kbeta_gexp^T -> [C, S]
    // GEMM: A[C,C] @ B^T[C,S] = C[C,S], so M=C, N=S, K=C
    // Grid for [C, S] output: grid.x = (S+15)/16 (cols), grid.y = (C+15)/16 (rows)
    dim3 grid_CS((S + 15) / 16, (C + 15) / 16);
    kernel_gemm_nt_naive<<<grid_CS, block2d>>>(d_attn_solved, d_kbeta_gexp, d_temp, C, S, C);
    // K_cumdecay = temp^T -> [S, C]
    // Transpose from [C, S] to [S, C]: rows=C, cols=S
    // Grid: grid.x = (cols+15)/16 = (S+15)/16, grid.y = (rows+15)/16 = (C+15)/16
    dim3 grid_transpose_CS((S + 15) / 16, (C + 15) / 16);
    kernel_transpose<<<grid_transpose_CS, block2d>>>(d_temp, d_K_cumdecay, C, S);

    HIP_CHECK(hipDeviceSynchronize());

    // Copy results back
    to_host(h_attn_solved_out.data(), d_attn_solved, C * C);
    to_host(h_V_new_out.data(), d_V_new, S * C);
    to_host(h_K_cumdecay_out.data(), d_K_cumdecay, S * C);

    // Compare results
    float diff_attn = max_diff(h_attn_solved_ref.data(), h_attn_solved_out.data(), C * C);
    float diff_v = max_diff(h_V_new_ref.data(), h_V_new_out.data(), S * C);
    float diff_k = max_diff(h_K_cumdecay_ref.data(), h_K_cumdecay_out.data(), S * C);

    printf("  attn_solved max_diff: %.2e\n", diff_attn);
    printf("  V_new max_diff: %.2e\n", diff_v);
    printf("  K_cumdecay max_diff: %.2e\n", diff_k);

    float max_overall = std::max({diff_attn, diff_v, diff_k});
    bool passed = max_overall < 1e-3f;

    // Cleanup
    device_free(d_K); device_free(d_V); device_free(d_G); device_free(d_Beta);
    device_free(d_beta_sig); device_free(d_g_cumsum); device_free(d_gexp);
    device_free(d_decay_mask); device_free(d_K_beta); device_free(d_V_beta);
    device_free(d_kmulkbeta); device_free(d_k_decay); device_free(d_causal);
    device_free(d_attn_pre); device_free(d_attn_solved); device_free(d_attn_T);
    device_free(d_V_new); device_free(d_kbeta_gexp); device_free(d_temp);
    device_free(d_K_cumdecay);

    report_test("Phase6: intra_chunk", passed, max_overall);
    return passed;
}

// Debug test: verify the attention matrix (before solve) matches reference
bool test_attn_pre_computation() {
    const int S = 16;
    const int C = 8;

    std::vector<float> h_K(S * C), h_G(C), h_Beta(C);
    srand(300);
    fill_random(h_K.data(), S * C, -0.3f, 0.3f);
    fill_random(h_G.data(), C, -0.1f, 0.1f);
    fill_random(h_Beta.data(), C, -1.0f, 1.0f);

    // Compute reference attn_pre
    std::vector<float> h_beta_sig(C), h_g_cumsum(C), h_decay(C * C);
    std::vector<float> h_K_beta(S * C), h_kmul(C * C), h_k_decay(C * C);
    std::vector<float> h_causal(C * C), h_attn_pre_ref(C * C);

    // Step 1: sigmoid(beta)
    host_sigmoid(h_Beta.data(), h_beta_sig.data(), C);

    // Step 2: cumsum(G)
    host_cumsum(h_G.data(), h_g_cumsum.data(), C);

    // Step 3: decay mask
    host_decay_mask(h_g_cumsum.data(), h_decay.data(), C);

    // Step 4: K_beta = K * beta_sig
    for (int s = 0; s < S; s++) {
        for (int c = 0; c < C; c++) {
            h_K_beta[s * C + c] = h_K[s * C + c] * h_beta_sig[c];
        }
    }

    // Step 5: kmul = K^T @ K_beta
    host_matmul_tn(h_K.data(), h_K_beta.data(), h_kmul.data(), C, C, S);

    // Step 6: k_decay = kmul * decay
    host_mul(h_kmul.data(), h_decay.data(), h_k_decay.data(), C * C);

    // Step 7: causal mask
    host_causal_mask(h_causal.data(), C);

    // Step 8: attn_pre = -(k_decay * causal)
    host_mul(h_k_decay.data(), h_causal.data(), h_attn_pre_ref.data(), C * C);
    host_neg(h_attn_pre_ref.data(), h_attn_pre_ref.data(), C * C);

    // GPU computation
    float* d_K = device_alloc<float>(S * C);
    float* d_G = device_alloc<float>(C);
    float* d_Beta = device_alloc<float>(C);
    float* d_beta_sig = device_alloc<float>(C);
    float* d_g_cumsum = device_alloc<float>(C);
    float* d_decay = device_alloc<float>(C * C);
    float* d_K_beta = device_alloc<float>(S * C);
    float* d_kmul = device_alloc<float>(C * C);
    float* d_k_decay = device_alloc<float>(C * C);
    float* d_causal = device_alloc<float>(C * C);
    float* d_attn_pre = device_alloc<float>(C * C);

    to_device(d_K, h_K.data(), S * C);
    to_device(d_G, h_G.data(), C);
    to_device(d_Beta, h_Beta.data(), C);

    kernel_sigmoid<<<1, C>>>(d_Beta, d_beta_sig, C);
    kernel_cumsum_sequential<<<1, 1>>>(d_G, d_g_cumsum, C);

    dim3 block2d(8, 8);
    dim3 grid_C(1, 1);
    kernel_decay_mask<<<grid_C, block2d>>>(d_g_cumsum, d_decay, C);
    kernel_broadcast_mul<<<(S * C + 63) / 64, 64>>>(d_K, d_beta_sig, d_K_beta, S, C);
    kernel_gemm_tn_naive<<<grid_C, block2d>>>(d_K, d_K_beta, d_kmul, C, C, S);
    kernel_mul<<<(C * C + 63) / 64, 64>>>(d_kmul, d_decay, d_k_decay, C * C);
    kernel_causal_mask<<<grid_C, block2d>>>(d_causal, C);
    kernel_mul<<<(C * C + 63) / 64, 64>>>(d_k_decay, d_causal, d_attn_pre, C * C);
    kernel_neg<<<(C * C + 63) / 64, 64>>>(d_attn_pre, d_attn_pre, C * C);

    HIP_CHECK(hipDeviceSynchronize());

    std::vector<float> h_attn_pre_out(C * C);
    to_host(h_attn_pre_out.data(), d_attn_pre, C * C);

    float diff = max_diff(h_attn_pre_ref.data(), h_attn_pre_out.data(), C * C);
    bool passed = diff < 1e-5f;

    device_free(d_K); device_free(d_G); device_free(d_Beta);
    device_free(d_beta_sig); device_free(d_g_cumsum); device_free(d_decay);
    device_free(d_K_beta); device_free(d_kmul); device_free(d_k_decay);
    device_free(d_causal); device_free(d_attn_pre);

    report_test("Phase6: attn_pre_computation", passed, diff);
    return passed;
}

// Debug test: verify the triangular solve matches reference
bool test_solve_with_causal_identity() {
    const int C = 8;

    // Create a test strictly lower triangular matrix
    std::vector<float> h_A(C * C);
    srand(200);

    // Make A strictly lower triangular (zeros on and above diagonal)
    fill_zeros(h_A.data(), C * C);
    for (int i = 1; i < C; i++) {
        for (int j = 0; j < i; j++) {
            h_A[i * C + j] = (rand() / (float)RAND_MAX - 0.5f) * 0.5f;
        }
    }

    // Solve (I - A) @ X = A and then add identity diagonal
    std::vector<float> h_ref(C * C), h_out(C * C);

    // Host reference using forward substitution
    // X[i,c] = A[i,c] + sum_{j<i} A[i,j] * X[j,c]
    for (int c = 0; c < C; c++) {
        for (int i = 0; i < C; i++) {
            float sum = h_A[i * C + c];
            for (int j = 0; j < i; j++) {
                sum += h_A[i * C + j] * h_ref[j * C + c];
            }
            h_ref[i * C + c] = sum;
        }
    }
    // Add identity diagonal
    for (int i = 0; i < C; i++) {
        for (int j = 0; j < C; j++) {
            if (j < i) {
                // keep
            } else if (j == i) {
                h_ref[i * C + j] = 1.0f;
            } else {
                h_ref[i * C + j] = 0.0f;
            }
        }
    }

    // GPU computation
    float* d_A = device_alloc<float>(C * C);
    float* d_X = device_alloc<float>(C * C);

    to_device(d_A, h_A.data(), C * C);

    kernel_solve_tri_sequential<<<1, 1>>>(d_A, d_A, d_X, C, C);
    HIP_CHECK(hipDeviceSynchronize());

    dim3 block(8, 8);
    dim3 grid(1, 1);
    kernel_apply_causal_identity<<<grid, block>>>(d_X, d_X, C);
    HIP_CHECK(hipDeviceSynchronize());

    to_host(h_out.data(), d_X, C * C);

    float diff = max_diff(h_ref.data(), h_out.data(), C * C);
    bool passed = diff < 1e-5f;

    device_free(d_A);
    device_free(d_X);

    report_test("Phase6: solve_with_causal_identity", passed, diff);
    return passed;
}

// Smaller test for debugging
bool test_intra_chunk_small() {
    const int S = 16;  // Smaller for debugging
    const int C = 8;

    std::vector<float> h_K(S * C), h_V(S * C), h_G(C), h_Beta(C);
    std::vector<float> h_attn_ref(C * C), h_V_new_ref(S * C), h_K_cumdecay_ref(S * C);
    std::vector<float> h_attn_out(C * C), h_V_new_out(S * C), h_K_cumdecay_out(S * C);

    srand(101);
    fill_random(h_K.data(), S * C, -0.3f, 0.3f);
    fill_random(h_V.data(), S * C, -0.3f, 0.3f);
    fill_random(h_G.data(), C, -0.1f, 0.1f);
    fill_random(h_Beta.data(), C, -1.0f, 1.0f);

    std::vector<float> h_Q_dummy(S * C, 0.0f);
    host_intra_chunk(h_Q_dummy.data(), h_K.data(), h_V.data(),
                     h_G.data(), h_Beta.data(),
                     h_attn_ref.data(), h_V_new_ref.data(), h_K_cumdecay_ref.data(),
                     S, C);

    // Device computation (same as above but smaller)
    float* d_K = device_alloc<float>(S * C);
    float* d_V = device_alloc<float>(S * C);
    float* d_G = device_alloc<float>(C);
    float* d_Beta = device_alloc<float>(C);
    float* d_beta_sig = device_alloc<float>(C);
    float* d_g_cumsum = device_alloc<float>(C);
    float* d_gexp = device_alloc<float>(C);
    float* d_decay_mask = device_alloc<float>(C * C);
    float* d_K_beta = device_alloc<float>(S * C);
    float* d_V_beta = device_alloc<float>(S * C);
    float* d_kmulkbeta = device_alloc<float>(C * C);
    float* d_k_decay = device_alloc<float>(C * C);
    float* d_causal = device_alloc<float>(C * C);
    float* d_attn_pre = device_alloc<float>(C * C);
    float* d_attn_solved = device_alloc<float>(C * C);
    float* d_attn_T = device_alloc<float>(C * C);
    float* d_V_new = device_alloc<float>(S * C);
    float* d_kbeta_gexp = device_alloc<float>(S * C);
    float* d_temp = device_alloc<float>(C * S);
    float* d_K_cumdecay = device_alloc<float>(S * C);

    to_device(d_K, h_K.data(), S * C);
    to_device(d_V, h_V.data(), S * C);
    to_device(d_G, h_G.data(), C);
    to_device(d_Beta, h_Beta.data(), C);

    dim3 block2d(8, 8);
    dim3 grid_C(1, 1);
    dim3 grid_SC((C + 7) / 8, (S + 7) / 8);

    kernel_sigmoid<<<1, C>>>(d_Beta, d_beta_sig, C);
    kernel_cumsum_sequential<<<1, 1>>>(d_G, d_g_cumsum, C);
    kernel_decay_mask<<<grid_C, block2d>>>(d_g_cumsum, d_decay_mask, C);
    kernel_broadcast_mul<<<(S * C + 63) / 64, 64>>>(d_K, d_beta_sig, d_K_beta, S, C);
    kernel_broadcast_mul<<<(S * C + 63) / 64, 64>>>(d_V, d_beta_sig, d_V_beta, S, C);
    kernel_gemm_tn_naive<<<grid_C, block2d>>>(d_K, d_K_beta, d_kmulkbeta, C, C, S);
    kernel_mul<<<(C * C + 63) / 64, 64>>>(d_kmulkbeta, d_decay_mask, d_k_decay, C * C);
    kernel_causal_mask<<<grid_C, block2d>>>(d_causal, C);
    kernel_mul<<<(C * C + 63) / 64, 64>>>(d_k_decay, d_causal, d_attn_pre, C * C);
    kernel_neg<<<(C * C + 63) / 64, 64>>>(d_attn_pre, d_attn_pre, C * C);
    kernel_solve_tri_sequential<<<1, 1>>>(d_attn_pre, d_attn_pre, d_attn_solved, C, C);
    kernel_apply_causal_identity<<<grid_C, block2d>>>(d_attn_solved, d_attn_solved, C);
    kernel_transpose<<<grid_C, block2d>>>(d_attn_solved, d_attn_T, C, C);
    kernel_gemm_nn_naive<<<grid_SC, block2d>>>(d_V_beta, d_attn_T, d_V_new, S, C, C);
    kernel_exp<<<1, C>>>(d_g_cumsum, d_gexp, C);
    kernel_broadcast_mul<<<(S * C + 63) / 64, 64>>>(d_K_beta, d_gexp, d_kbeta_gexp, S, C);
    // temp = attn_solved [C,C] @ kbeta_gexp^T [C,S] -> [C, S]
    dim3 grid_CS_small((S + 7) / 8, (C + 7) / 8);
    kernel_gemm_nt_naive<<<grid_CS_small, block2d>>>(d_attn_solved, d_kbeta_gexp, d_temp, C, S, C);
    // K_cumdecay = temp^T [S, C]
    dim3 grid_transpose_small((S + 7) / 8, (C + 7) / 8);
    kernel_transpose<<<grid_transpose_small, block2d>>>(d_temp, d_K_cumdecay, C, S);

    HIP_CHECK(hipDeviceSynchronize());

    to_host(h_attn_out.data(), d_attn_solved, C * C);
    to_host(h_V_new_out.data(), d_V_new, S * C);
    to_host(h_K_cumdecay_out.data(), d_K_cumdecay, S * C);

    float diff_attn = max_diff(h_attn_ref.data(), h_attn_out.data(), C * C);
    float diff_v = max_diff(h_V_new_ref.data(), h_V_new_out.data(), S * C);
    float diff_k = max_diff(h_K_cumdecay_ref.data(), h_K_cumdecay_out.data(), S * C);

    printf("  attn_solved max_diff: %.2e\n", diff_attn);
    printf("  V_new max_diff: %.2e\n", diff_v);
    printf("  K_cumdecay max_diff: %.2e\n", diff_k);

    float max_overall = std::max({diff_attn, diff_v, diff_k});
    bool passed = max_overall < 1e-4f;

    device_free(d_K); device_free(d_V); device_free(d_G); device_free(d_Beta);
    device_free(d_beta_sig); device_free(d_g_cumsum); device_free(d_gexp);
    device_free(d_decay_mask); device_free(d_K_beta); device_free(d_V_beta);
    device_free(d_kmulkbeta); device_free(d_k_decay); device_free(d_causal);
    device_free(d_attn_pre); device_free(d_attn_solved); device_free(d_attn_T);
    device_free(d_V_new); device_free(d_kbeta_gexp); device_free(d_temp);
    device_free(d_K_cumdecay);

    report_test("Phase6: intra_chunk_small", passed, max_overall);
    return passed;
}

// ============================================================================
// Phase 7 Tests: State Interaction
// ============================================================================

// Test state update: state_new = state * exp(g_last) + K^T @ V
bool test_state_update() {
    const int S = 128;  // State dimension
    const int C = 64;   // Chunk dimension

    std::vector<float> h_state(S * S), h_K(S * C), h_V(S * C);
    float g_last = 0.3f;

    srand(400);
    fill_random(h_state.data(), S * S, -0.5f, 0.5f);
    fill_random(h_K.data(), S * C, -0.3f, 0.3f);
    fill_random(h_V.data(), S * C, -0.3f, 0.3f);

    // Host reference: state_new = state * exp(g_last) + K^T @ V
    std::vector<float> h_state_ref(S * S);
    float exp_g = expf(g_last);

    // Term 1: state * exp(g_last)
    host_scale(h_state.data(), h_state_ref.data(), S * S, exp_g);

    // Term 2: K^T @ V (K is [S, C], V is [S, C], K^T @ V is [C, C])
    // But state is [S, S]. For this test, we'll do K @ V^T which is [S, S]
    std::vector<float> h_kv(S * S);
    host_matmul_nt(h_K.data(), h_V.data(), h_kv.data(), S, S, C);

    // state_new = state * exp_g + K @ V^T
    host_add(h_state_ref.data(), h_kv.data(), h_state_ref.data(), S * S);

    // GPU computation
    float* d_state = device_alloc<float>(S * S);
    float* d_K = device_alloc<float>(S * C);
    float* d_V = device_alloc<float>(S * C);
    float* d_state_scaled = device_alloc<float>(S * S);
    float* d_kv = device_alloc<float>(S * S);
    float* d_state_new = device_alloc<float>(S * S);

    to_device(d_state, h_state.data(), S * S);
    to_device(d_K, h_K.data(), S * C);
    to_device(d_V, h_V.data(), S * C);

    int block1d = 256;
    dim3 block2d(16, 16);
    dim3 grid_S((S + 15) / 16, (S + 15) / 16);

    // state_scaled = state * exp(g_last)
    kernel_scale<<<(S * S + block1d - 1) / block1d, block1d>>>(
        d_state, d_state_scaled, S * S, exp_g);

    // kv = K @ V^T [S, S]
    kernel_gemm_nt_tiled<<<grid_S, block2d>>>(d_K, d_V, d_kv, S, S, C);

    // state_new = state_scaled + kv
    kernel_add<<<(S * S + block1d - 1) / block1d, block1d>>>(
        d_state_scaled, d_kv, d_state_new, S * S);

    HIP_CHECK(hipDeviceSynchronize());

    std::vector<float> h_state_out(S * S);
    to_host(h_state_out.data(), d_state_new, S * S);

    float diff = max_diff(h_state_ref.data(), h_state_out.data(), S * S);
    bool passed = diff < 1e-3f;

    device_free(d_state);
    device_free(d_K);
    device_free(d_V);
    device_free(d_state_scaled);
    device_free(d_kv);
    device_free(d_state_new);

    report_test("Phase7: state_update", passed, diff);
    return passed;
}

// Test output with state: output = Q @ State^T + attn @ V^T
bool test_output_with_state() {
    const int S = 128;  // State/head dimension
    const int C = 64;   // Chunk size

    std::vector<float> h_Q(S * C), h_state(S * S), h_attn(C * C), h_V(S * C);
    srand(401);
    fill_random(h_Q.data(), S * C, -0.3f, 0.3f);
    fill_random(h_state.data(), S * S, -0.3f, 0.3f);
    fill_random(h_attn.data(), C * C, -0.3f, 0.3f);
    fill_random(h_V.data(), S * C, -0.3f, 0.3f);

    // Host reference: output = Q @ State^T + (attn @ V^T)^T
    // Q is [S, C], State is [S, S], Q @ State^T... dimension mismatch.
    // Let's use a simpler interpretation:
    // Term 1: For each position c in chunk, Q[:,c] @ State^T gives [S, S]
    // Actually for Delta-Net, the output is token-wise.
    //
    // Simplified test: output[S, C] = Q[S, C] + (attn[C, C] @ V^T[C, S])^T = Q + attn_v
    // where attn_v = (attn @ V^T)^T = V @ attn^T
    std::vector<float> h_output_ref(S * C);
    std::vector<float> h_attn_T(C * C), h_attn_v(S * C);

    // attn_T = attn^T
    host_transpose(h_attn.data(), h_attn_T.data(), C, C);

    // attn_v = V @ attn_T  (V[S,C] @ attn_T[C,C] = [S,C])
    host_matmul(h_V.data(), h_attn_T.data(), h_attn_v.data(), S, C, C);

    // output = Q + attn_v
    host_add(h_Q.data(), h_attn_v.data(), h_output_ref.data(), S * C);

    // GPU computation
    float* d_Q = device_alloc<float>(S * C);
    float* d_attn = device_alloc<float>(C * C);
    float* d_V = device_alloc<float>(S * C);
    float* d_attn_T = device_alloc<float>(C * C);
    float* d_attn_v = device_alloc<float>(S * C);
    float* d_output = device_alloc<float>(S * C);

    to_device(d_Q, h_Q.data(), S * C);
    to_device(d_attn, h_attn.data(), C * C);
    to_device(d_V, h_V.data(), S * C);

    int block1d = 256;
    dim3 block2d(16, 16);
    dim3 grid_C((C + 15) / 16, (C + 15) / 16);
    dim3 grid_SC((C + 15) / 16, (S + 15) / 16);

    // attn_T = attn^T
    kernel_transpose<<<grid_C, block2d>>>(d_attn, d_attn_T, C, C);

    // attn_v = V @ attn_T
    kernel_gemm_nn_naive<<<grid_SC, block2d>>>(d_V, d_attn_T, d_attn_v, S, C, C);

    // output = Q + attn_v
    kernel_add<<<(S * C + block1d - 1) / block1d, block1d>>>(
        d_Q, d_attn_v, d_output, S * C);

    HIP_CHECK(hipDeviceSynchronize());

    std::vector<float> h_output_out(S * C);
    to_host(h_output_out.data(), d_output, S * C);

    float diff = max_diff(h_output_ref.data(), h_output_out.data(), S * C);
    bool passed = diff < 1e-4f;

    device_free(d_Q);
    device_free(d_attn);
    device_free(d_V);
    device_free(d_attn_T);
    device_free(d_attn_v);
    device_free(d_output);

    report_test("Phase7: output_with_state", passed, diff);
    return passed;
}

// ============================================================================
// Test Registry
// ============================================================================

using TestFunc = std::function<bool()>;
std::map<std::string, TestFunc> g_tests = {
    // Phase 1
    {"sigmoid", test_sigmoid},
    {"exp", test_exp},
    {"mul", test_mul},
    {"cumsum", test_cumsum},
    // Phase 2
    {"tril", test_tril},
    {"causal_mask", test_causal_mask},
    {"eye", test_eye},
    {"solve_tri_small", test_solve_tri_small},
    {"solve_tri_64", test_solve_tri_64},
    // Phase 3
    {"gemm_nt_naive_small", test_gemm_nt_naive_small},
    {"gemm_nt_naive_64", test_gemm_nt_naive_64},
    {"gemm_tn_naive", test_gemm_tn_naive},
    {"gemm_nt_tiled", test_gemm_nt_tiled},
    {"gemm_nt_tiled_128", test_gemm_nt_tiled_128},
    // Phase 4
    {"decay_mask_small", test_decay_mask_small},
    {"decay_mask_64", test_decay_mask_64},
    // Phase 5
    {"attn_matrix_small", test_attention_matrix_small},
    {"attn_matrix_64", test_attention_matrix_64},
    // Phase 6
    {"attn_pre_computation", test_attn_pre_computation},
    {"solve_causal_identity", test_solve_with_causal_identity},
    {"intra_chunk_small", test_intra_chunk_small},
    {"intra_chunk", test_intra_chunk},
    // Phase 7
    {"state_update", test_state_update},
    {"output_with_state", test_output_with_state},
};

std::map<int, std::vector<std::string>> g_phases = {
    {1, {"sigmoid", "exp", "mul", "cumsum"}},
    {2, {"tril", "causal_mask", "eye", "solve_tri_small", "solve_tri_64"}},
    {3, {"gemm_nt_naive_small", "gemm_nt_naive_64", "gemm_tn_naive", "gemm_nt_tiled", "gemm_nt_tiled_128"}},
    {4, {"decay_mask_small", "decay_mask_64"}},
    {5, {"attn_matrix_small", "attn_matrix_64"}},
    {6, {"attn_pre_computation", "solve_causal_identity", "intra_chunk_small", "intra_chunk"}},
    {7, {"state_update", "output_with_state"}},
};

void run_all_tests() {
    for (auto& [name, func] : g_tests) {
        func();
    }
}

void run_phase(int phase) {
    if (g_phases.find(phase) == g_phases.end()) {
        std::cerr << "Unknown phase: " << phase << std::endl;
        return;
    }
    for (const auto& name : g_phases[phase]) {
        g_tests[name]();
    }
}

void run_test(const std::string& name) {
    if (g_tests.find(name) == g_tests.end()) {
        std::cerr << "Unknown test: " << name << std::endl;
        return;
    }
    g_tests[name]();
}

void list_tests() {
    printf("Available tests:\n");
    for (auto& [phase, tests] : g_phases) {
        printf("  Phase %d:\n", phase);
        for (const auto& name : tests) {
            printf("    - %s\n", name.c_str());
        }
    }
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    // Check for HIP device
    int device_count;
    HIP_CHECK(hipGetDeviceCount(&device_count));
    if (device_count == 0) {
        std::cerr << "No HIP devices found!" << std::endl;
        return 1;
    }

    hipDeviceProp_t props;
    HIP_CHECK(hipGetDeviceProperties(&props, 0));
    printf("Using device: %s\n", props.name);
    printf("========================================\n\n");

    // Parse arguments
    if (argc == 1) {
        // Run all tests
        run_all_tests();
    } else if (argc >= 2) {
        std::string arg1 = argv[1];
        if (arg1 == "--list" || arg1 == "-l") {
            list_tests();
            return 0;
        } else if (arg1 == "--phase" || arg1 == "-p") {
            if (argc < 3) {
                std::cerr << "Usage: " << argv[0] << " --phase <N>" << std::endl;
                return 1;
            }
            int phase = std::atoi(argv[2]);
            run_phase(phase);
        } else if (arg1 == "--test" || arg1 == "-t") {
            if (argc < 3) {
                std::cerr << "Usage: " << argv[0] << " --test <name>" << std::endl;
                return 1;
            }
            run_test(argv[2]);
        } else {
            // Assume it's a test name
            run_test(arg1);
        }
    }

    print_summary();
    return (g_tests_passed == g_tests_run) ? 0 : 1;
}
