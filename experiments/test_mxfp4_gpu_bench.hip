// GPU microbenchmark: MXFP4 table lookup vs compute
// Compile: hipcc -O3 --offload-arch=gfx906 -o test_mxfp4_gpu_bench test_mxfp4_gpu_bench.hip
// Run: HSA_OVERRIDE_GFX_VERSION=9.0.6 ./test_mxfp4_gpu_bench

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>

#define ITERATIONS 1000000
#define THREADS 256
#define BLOCKS 256

// Original table (in constant memory)
__device__ __constant__ int8_t kvalues_mxfp4[16] = {
    0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12
};

// Original get_int_from_table_16 using v_perm (6 perm ops)
static __device__ __forceinline__ int2 get_int_from_table_16_perm(
    const uint32_t q4, const int8_t * values) {

    uint32_t values_u32[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        values_u32[i] = reinterpret_cast<const uint32_t*>(values)[i];
    }

    uint32_t q_even = (q4 & 0x0f0f0f0f);
    uint32_t q_odd  = (q4 >> 4) & 0x0f0f0f0f;

    uint32_t mask_even = (q_even >> 3) * 0xff;
    uint32_t mask_odd  = (q_odd >> 3)  * 0xff;
    mask_even = (mask_even & 0x0c0c0c0c) | 0x03020100;
    mask_odd  = (mask_odd  & 0x0c0c0c0c) | 0x03020100;

    uint32_t v_even_low  = __builtin_amdgcn_perm(values_u32[1], values_u32[0], q_even & 0x07070707);
    uint32_t v_odd_low   = __builtin_amdgcn_perm(values_u32[1], values_u32[0], q_odd  & 0x07070707);
    uint32_t v_even_high = __builtin_amdgcn_perm(values_u32[3], values_u32[2], q_even & 0x07070707);
    uint32_t v_odd_high  = __builtin_amdgcn_perm(values_u32[3], values_u32[2], q_odd  & 0x07070707);
    uint32_t res_x = __builtin_amdgcn_perm(v_even_high, v_even_low, mask_even);
    uint32_t res_y = __builtin_amdgcn_perm(v_odd_high, v_odd_low, mask_odd);

    return make_int2(res_x, res_y);
}

// Compute version - process one nibble
static __device__ __forceinline__ int8_t compute_mxfp4_value(int idx) {
    int sign = (idx >> 3) & 1;
    int m = idx & 0x7;

    int is_high = (m >> 2) & 1;
    int m_low = m & 3;
    int is_3 = (m_low & 1) & (m_low >> 1);

    int mag_high = 4 + 2*m_low + 2*is_3;
    int mag = m * (1 - is_high) + mag_high * is_high;

    return (int8_t)(mag * (1 - 2*sign));
}

// Compute version that processes all 8 nibbles from aux_q4
static __device__ __forceinline__ int2 get_int_from_compute_mxfp4(uint32_t q4) {
    int8_t results[8];

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int byte = (q4 >> (i * 8)) & 0xff;
        int lo_nibble = byte & 0xf;
        int hi_nibble = (byte >> 4) & 0xf;
        results[i]     = compute_mxfp4_value(lo_nibble);  // even positions
        results[i + 4] = compute_mxfp4_value(hi_nibble);  // odd positions
    }

    // Pack into int2 (same layout as table version)
    uint32_t res_x = ((uint8_t)results[0]) |
                     ((uint8_t)results[1] << 8) |
                     ((uint8_t)results[2] << 16) |
                     ((uint8_t)results[3] << 24);
    uint32_t res_y = ((uint8_t)results[4]) |
                     ((uint8_t)results[5] << 8) |
                     ((uint8_t)results[6] << 16) |
                     ((uint8_t)results[7] << 24);

    return make_int2(res_x, res_y);
}

// Benchmark kernel using table lookup
__global__ void bench_table_lookup(uint32_t* input, int2* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int2 acc = make_int2(0, 0);
    uint32_t val = input[idx % n];

    for (int i = 0; i < ITERATIONS; i++) {
        int2 v = get_int_from_table_16_perm(val ^ i, kvalues_mxfp4);
        acc.x += v.x;
        acc.y += v.y;
        val = val ^ v.x;  // Data dependency to prevent over-optimization
    }

    output[idx] = acc;
}

// Benchmark kernel using compute
__global__ void bench_compute(uint32_t* input, int2* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int2 acc = make_int2(0, 0);
    uint32_t val = input[idx % n];

    for (int i = 0; i < ITERATIONS; i++) {
        int2 v = get_int_from_compute_mxfp4(val ^ i);
        acc.x += v.x;
        acc.y += v.y;
        val = val ^ v.x;
    }

    output[idx] = acc;
}

// Correctness test kernel
__global__ void test_correctness(int* errors) {
    int idx = threadIdx.x;
    if (idx >= 256) return;

    // Test all possible byte values
    uint32_t test_val = idx | (idx << 8) | (idx << 16) | (idx << 24);

    int2 table_result = get_int_from_table_16_perm(test_val, kvalues_mxfp4);
    int2 compute_result = get_int_from_compute_mxfp4(test_val);

    if (table_result.x != compute_result.x || table_result.y != compute_result.y) {
        atomicAdd(errors, 1);
        if (idx < 16) {
            printf("Mismatch at idx=%d: table=(%08x,%08x) compute=(%08x,%08x)\n",
                   idx, table_result.x, table_result.y,
                   compute_result.x, compute_result.y);
        }
    }
}

int main() {
    printf("=== MXFP4 Table Lookup vs Compute Benchmark ===\n");
    printf("Iterations per thread: %d\n", ITERATIONS);
    printf("Total threads: %d\n\n", THREADS * BLOCKS);

    // Allocate
    uint32_t* d_input;
    int2* d_output;
    int* d_errors;

    hipMalloc(&d_input, THREADS * BLOCKS * sizeof(uint32_t));
    hipMalloc(&d_output, THREADS * BLOCKS * sizeof(int2));
    hipMalloc(&d_errors, sizeof(int));

    // Initialize with random-ish data
    uint32_t* h_input = new uint32_t[THREADS * BLOCKS];
    for (int i = 0; i < THREADS * BLOCKS; i++) {
        h_input[i] = i * 0x12345678;
    }
    hipMemcpy(d_input, h_input, THREADS * BLOCKS * sizeof(uint32_t), hipMemcpyHostToDevice);

    // Test correctness first
    printf("Testing correctness...\n");
    int h_errors = 0;
    hipMemcpy(d_errors, &h_errors, sizeof(int), hipMemcpyHostToDevice);
    test_correctness<<<1, 256>>>(d_errors);
    hipDeviceSynchronize();
    hipMemcpy(&h_errors, d_errors, sizeof(int), hipMemcpyDeviceToHost);

    if (h_errors > 0) {
        printf("FAILED: %d mismatches!\n", h_errors);
        return 1;
    }
    printf("Correctness: PASSED\n\n");

    // Warmup
    bench_table_lookup<<<BLOCKS, THREADS>>>(d_input, d_output, THREADS * BLOCKS);
    bench_compute<<<BLOCKS, THREADS>>>(d_input, d_output, THREADS * BLOCKS);
    hipDeviceSynchronize();

    // Benchmark table lookup
    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);

    hipEventRecord(start);
    bench_table_lookup<<<BLOCKS, THREADS>>>(d_input, d_output, THREADS * BLOCKS);
    hipEventRecord(stop);
    hipEventSynchronize(stop);

    float table_ms;
    hipEventElapsedTime(&table_ms, start, stop);

    // Benchmark compute
    hipEventRecord(start);
    bench_compute<<<BLOCKS, THREADS>>>(d_input, d_output, THREADS * BLOCKS);
    hipEventRecord(stop);
    hipEventSynchronize(stop);

    float compute_ms;
    hipEventElapsedTime(&compute_ms, start, stop);

    // Results
    double total_ops = (double)ITERATIONS * THREADS * BLOCKS;

    printf("Results:\n");
    printf("  Table lookup: %.2f ms (%.2f Gops/s)\n",
           table_ms, total_ops / table_ms / 1e6);
    printf("  Compute:      %.2f ms (%.2f Gops/s)\n",
           compute_ms, total_ops / compute_ms / 1e6);
    printf("\n");
    printf("  Speedup: %.2fx %s\n",
           table_ms > compute_ms ? table_ms / compute_ms : compute_ms / table_ms,
           table_ms > compute_ms ? "(compute faster)" : "(table faster)");

    // Cleanup
    hipFree(d_input);
    hipFree(d_output);
    hipFree(d_errors);
    delete[] h_input;

    hipEventDestroy(start);
    hipEventDestroy(stop);

    return 0;
}
