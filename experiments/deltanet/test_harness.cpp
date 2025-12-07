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
    int row = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        out[row * n + col] = (col < row) ? 1.0f : 0.0f;
    }
}

__global__ void kernel_eye(float* __restrict__ out, int n) {
    int row = blockIdx.y;
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
};

std::map<int, std::vector<std::string>> g_phases = {
    {1, {"sigmoid", "exp", "mul", "cumsum"}},
    {2, {"tril", "causal_mask", "eye", "solve_tri_small", "solve_tri_64"}},
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
