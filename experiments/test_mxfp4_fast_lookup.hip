// Test kernel for MXFP4 fast lookup optimization
// Compares 6 v_perm (original) vs 4 v_perm (optimized) approach

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>

// Original lookup table
__device__ __constant__ int8_t kvalues_mxfp4[16] = {
    0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12
};

// ============================================================================
// ORIGINAL: 6 v_perm implementation
// ============================================================================
__device__ __forceinline__ int2 lookup_original(uint32_t q4) {
    const uint32_t *values = (const uint32_t *)kvalues_mxfp4;

    const uint32_t q_even = q4;
    const uint32_t q_odd  = (q4 >> 4);

    // 4 v_perm for table lookups
    uint32_t v_even_low  = __builtin_amdgcn_perm(values[1], values[0], q_even & 0x07070707);
    uint32_t v_odd_low   = __builtin_amdgcn_perm(values[1], values[0], q_odd  & 0x07070707);
    uint32_t v_even_high = __builtin_amdgcn_perm(values[3], values[2], q_even & 0x07070707);
    uint32_t v_odd_high  = __builtin_amdgcn_perm(values[3], values[2], q_odd  & 0x07070707);

    // 2 v_perm for selection
    uint32_t mask_even = 0x03020100 | ((q_even & 0x08080808) >> 1);
    uint32_t mask_odd  = 0x03020100 | ((q_odd  & 0x08080808) >> 1);
    uint32_t res_x = __builtin_amdgcn_perm(v_even_high, v_even_low, mask_even);
    uint32_t res_y = __builtin_amdgcn_perm(v_odd_high,  v_odd_low,  mask_odd);

    return make_int2(res_x, res_y);
}

// ============================================================================
// OPTIMIZED: 4 v_perm + BFI selection
// ============================================================================
__device__ __forceinline__ int2 lookup_fast(uint32_t q4) {
    // Separate positive and negative 8-entry tables
    const uint32_t pos_lo = 0x03020100;  // {0, 1, 2, 3}
    const uint32_t pos_hi = 0x0c080604;  // {4, 6, 8, 12}
    const uint32_t neg_lo = 0xfdfeff00;  // {0, -1, -2, -3}
    const uint32_t neg_hi = 0xf4f8fafc;  // {-4, -6, -8, -12}
    const uint32_t expand = 0x0000FF00;  // sign bit expander: 0->0x00, 1->0xFF

    const uint32_t q_even = q4;
    const uint32_t q_odd  = q4 >> 4;
    const uint32_t mag_idx_even = q_even & 0x07070707;
    const uint32_t mag_idx_odd  = q_odd  & 0x07070707;

    // Lookup positive values: 2 v_perm
    uint32_t pos_even = __builtin_amdgcn_perm(pos_hi, pos_lo, mag_idx_even);
    uint32_t pos_odd  = __builtin_amdgcn_perm(pos_hi, pos_lo, mag_idx_odd);

    // Lookup negative values: 2 v_perm
    uint32_t neg_even = __builtin_amdgcn_perm(neg_hi, neg_lo, mag_idx_even);
    uint32_t neg_odd  = __builtin_amdgcn_perm(neg_hi, neg_lo, mag_idx_odd);

    // Extract sign bits (bit 3 of each nibble)
    uint32_t sign_even = (q_even >> 3) & 0x01010101;
    uint32_t sign_odd  = (q_odd  >> 3) & 0x01010101;

    // Expand sign to byte mask: 2 v_perm
    uint32_t mask_even = __builtin_amdgcn_perm(expand, expand, sign_even);
    uint32_t mask_odd  = __builtin_amdgcn_perm(expand, expand, sign_odd);

    // BFI selection: result = (mask & neg) | (~mask & pos)
    // 2 v_bfi instructions
    uint32_t res_x, res_y;
    asm volatile("v_bfi_b32 %0, %1, %2, %3" : "=v"(res_x) : "v"(mask_even), "v"(neg_even), "v"(pos_even));
    asm volatile("v_bfi_b32 %0, %1, %2, %3" : "=v"(res_y) : "v"(mask_odd),  "v"(neg_odd),  "v"(pos_odd));

    return make_int2(res_x, res_y);
}

// ============================================================================
// Test kernel: verify correctness
// ============================================================================
__global__ void test_correctness(uint32_t* test_inputs, int2* results_orig, int2* results_fast, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    uint32_t q4 = test_inputs[idx];
    results_orig[idx] = lookup_original(q4);
    results_fast[idx] = lookup_fast(q4);
}

// ============================================================================
// Benchmark kernel: original (6 v_perm)
// ============================================================================
__global__ void bench_original(uint32_t* inputs, int2* outputs, int n, int iters) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    uint32_t q4 = inputs[idx];
    int2 result = make_int2(0, 0);

    #pragma unroll 1
    for (int i = 0; i < iters; i++) {
        int2 r = lookup_original(q4 ^ i);  // XOR to prevent optimization
        result.x += r.x;
        result.y += r.y;
    }

    outputs[idx] = result;
}

// ============================================================================
// Benchmark kernel: fast (4 v_perm)
// ============================================================================
__global__ void bench_fast(uint32_t* inputs, int2* outputs, int n, int iters) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    uint32_t q4 = inputs[idx];
    int2 result = make_int2(0, 0);

    #pragma unroll 1
    for (int i = 0; i < iters; i++) {
        int2 r = lookup_fast(q4 ^ i);  // XOR to prevent optimization
        result.x += r.x;
        result.y += r.y;
    }

    outputs[idx] = result;
}

// ============================================================================
// Host helpers
// ============================================================================
void check_hip(hipError_t err, const char* msg) {
    if (err != hipSuccess) {
        fprintf(stderr, "HIP Error: %s: %s\n", msg, hipGetErrorString(err));
        exit(1);
    }
}

int main() {
    printf("=== MXFP4 Fast Lookup Test (4 v_perm vs 6 v_perm) ===\n\n");

    // Test parameters
    const int N = 65536;  // Number of test values
    const int ITERS = 10000;  // Iterations per benchmark
    const int BLOCK_SIZE = 256;
    const int GRID_SIZE = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Allocate host memory
    uint32_t* h_inputs = new uint32_t[N];
    int2* h_results_orig = new int2[N];
    int2* h_results_fast = new int2[N];

    // Generate test inputs (all possible 4-nibble patterns + random)
    for (int i = 0; i < N; i++) {
        if (i < 65536) {
            h_inputs[i] = i;  // All 16-bit patterns (covers all nibble combos)
        } else {
            h_inputs[i] = rand();
        }
    }

    // Allocate device memory
    uint32_t* d_inputs;
    int2* d_results_orig;
    int2* d_results_fast;

    check_hip(hipMalloc(&d_inputs, N * sizeof(uint32_t)), "malloc inputs");
    check_hip(hipMalloc(&d_results_orig, N * sizeof(int2)), "malloc results_orig");
    check_hip(hipMalloc(&d_results_fast, N * sizeof(int2)), "malloc results_fast");

    check_hip(hipMemcpy(d_inputs, h_inputs, N * sizeof(uint32_t), hipMemcpyHostToDevice), "memcpy inputs");

    // ========================================================================
    // Test 1: Correctness
    // ========================================================================
    printf("[1] Testing correctness...\n");

    hipLaunchKernelGGL(test_correctness, dim3(GRID_SIZE), dim3(BLOCK_SIZE), 0, 0,
                       d_inputs, d_results_orig, d_results_fast, N);
    check_hip(hipDeviceSynchronize(), "correctness kernel");

    check_hip(hipMemcpy(h_results_orig, d_results_orig, N * sizeof(int2), hipMemcpyDeviceToHost), "memcpy orig");
    check_hip(hipMemcpy(h_results_fast, d_results_fast, N * sizeof(int2), hipMemcpyDeviceToHost), "memcpy fast");

    int errors = 0;
    for (int i = 0; i < N && errors < 10; i++) {
        if (h_results_orig[i].x != h_results_fast[i].x ||
            h_results_orig[i].y != h_results_fast[i].y) {
            printf("  MISMATCH at i=%d, input=0x%08x:\n", i, h_inputs[i]);
            printf("    Original: (0x%08x, 0x%08x)\n", h_results_orig[i].x, h_results_orig[i].y);
            printf("    Fast:     (0x%08x, 0x%08x)\n", h_results_fast[i].x, h_results_fast[i].y);
            errors++;
        }
    }

    if (errors == 0) {
        printf("  PASSED: All %d test cases match!\n\n", N);
    } else {
        printf("  FAILED: %d mismatches found\n\n", errors);
        return 1;
    }

    // ========================================================================
    // Test 2: Performance benchmark
    // ========================================================================
    printf("[2] Benchmarking (%d elements x %d iterations)...\n", N, ITERS);

    // Warmup
    hipLaunchKernelGGL(bench_original, dim3(GRID_SIZE), dim3(BLOCK_SIZE), 0, 0,
                       d_inputs, d_results_orig, N, 100);
    hipLaunchKernelGGL(bench_fast, dim3(GRID_SIZE), dim3(BLOCK_SIZE), 0, 0,
                       d_inputs, d_results_fast, N, 100);
    check_hip(hipDeviceSynchronize(), "warmup");

    // Benchmark original
    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);

    hipEventRecord(start);
    hipLaunchKernelGGL(bench_original, dim3(GRID_SIZE), dim3(BLOCK_SIZE), 0, 0,
                       d_inputs, d_results_orig, N, ITERS);
    hipEventRecord(stop);
    hipEventSynchronize(stop);

    float ms_orig;
    hipEventElapsedTime(&ms_orig, start, stop);

    // Benchmark fast
    hipEventRecord(start);
    hipLaunchKernelGGL(bench_fast, dim3(GRID_SIZE), dim3(BLOCK_SIZE), 0, 0,
                       d_inputs, d_results_fast, N, ITERS);
    hipEventRecord(stop);
    hipEventSynchronize(stop);

    float ms_fast;
    hipEventElapsedTime(&ms_fast, start, stop);

    // Results
    double ops_orig = (double)N * ITERS / (ms_orig / 1000.0);
    double ops_fast = (double)N * ITERS / (ms_fast / 1000.0);
    double speedup = ms_orig / ms_fast;

    printf("\n  Original (6 v_perm): %.2f ms  (%.2f Gops/s)\n", ms_orig, ops_orig / 1e9);
    printf("  Fast (4 v_perm):     %.2f ms  (%.2f Gops/s)\n", ms_fast, ops_fast / 1e9);
    printf("\n  Speedup: %.2fx\n", speedup);

    if (speedup > 1.0) {
        printf("  Result: FASTER by %.1f%%\n", (speedup - 1.0) * 100);
    } else {
        printf("  Result: SLOWER by %.1f%%\n", (1.0 - speedup) * 100);
    }

    // Cleanup
    hipFree(d_inputs);
    hipFree(d_results_orig);
    hipFree(d_results_fast);
    delete[] h_inputs;
    delete[] h_results_orig;
    delete[] h_results_fast;
    hipEventDestroy(start);
    hipEventDestroy(stop);

    printf("\n=== Done ===\n");
    return 0;
}
