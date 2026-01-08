// Compute-bound benchmark to reveal true perm savings
// The previous test was memory-bound, hiding the compute difference
//
// Key insight: In real MMQ kernel, we do MULTIPLE lookups per memory load
// and accumulate results. This makes compute the bottleneck.

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>

// All-negative specialized (2 perms)
__device__ __forceinline__ int2 neg_2perm(uint32_t q4) {
    const uint32_t neg_lo = 0xfdfeff00;
    const uint32_t neg_hi = 0xf4f8fafc;

    uint32_t idx_e = q4 & 0x07070707;
    uint32_t idx_o = (q4 >> 4) & 0x07070707;

    uint32_t res_x, res_y;
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(res_x) : "v"(neg_hi), "v"(neg_lo), "v"(idx_e));
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(res_y) : "v"(neg_hi), "v"(neg_lo), "v"(idx_o));

    return make_int2(res_x, res_y);
}

// Baseline 6-perm
__device__ __forceinline__ int2 baseline_6perm(uint32_t q4) {
    const uint32_t values0 = 0x03020100;
    const uint32_t values1 = 0x0c080604;
    const uint32_t values2 = 0xfdfeff00;
    const uint32_t values3 = 0xf4f8fafc;

    const uint32_t q_even = q4;
    const uint32_t q_odd  = q4 >> 4;

    uint32_t v_even_low, v_odd_low, v_even_high, v_odd_high;
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(v_even_low)  : "v"(values1), "v"(values0), "v"(q_even & 0x07070707));
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(v_odd_low)   : "v"(values1), "v"(values0), "v"(q_odd  & 0x07070707));
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(v_even_high) : "v"(values3), "v"(values2), "v"(q_even & 0x07070707));
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(v_odd_high)  : "v"(values3), "v"(values2), "v"(q_odd  & 0x07070707));

    const uint32_t mask_even = 0x03020100 | ((q_even & 0x08080808) >> 1);
    const uint32_t mask_odd  = 0x03020100 | ((q_odd  & 0x08080808) >> 1);

    uint32_t res_x, res_y;
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(res_x) : "v"(v_even_high), "v"(v_even_low), "v"(mask_even));
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(res_y) : "v"(v_odd_high),  "v"(v_odd_low),  "v"(mask_odd));

    return make_int2(res_x, res_y);
}

// ============================================================================
// COMPUTE-BOUND BENCHMARKS
// Each thread does MULTIPLE lookups and accumulates - simulating MMQ
// ============================================================================

// Baseline: 6 perms per lookup, N lookups per thread
template<int LOOKUPS_PER_THREAD>
__global__ void bench_baseline_compute(const uint32_t* __restrict__ data, int* __restrict__ out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Load once
    uint32_t q4 = data[idx];

    // Do multiple lookups (simulating processing a block)
    int acc_x = 0, acc_y = 0;
    #pragma unroll
    for (int i = 0; i < LOOKUPS_PER_THREAD; i++) {
        // Vary the input slightly to prevent compiler from optimizing away
        uint32_t q_varied = q4 ^ (i * 0x11111111);
        int2 result = baseline_6perm(q_varied);
        acc_x += result.x;
        acc_y += result.y;
    }

    out[idx] = acc_x + acc_y;
}

// Specialized: 2 perms per lookup, N lookups per thread
template<int LOOKUPS_PER_THREAD>
__global__ void bench_neg2perm_compute(const uint32_t* __restrict__ data, int* __restrict__ out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    uint32_t q4 = data[idx];

    int acc_x = 0, acc_y = 0;
    #pragma unroll
    for (int i = 0; i < LOOKUPS_PER_THREAD; i++) {
        uint32_t q_varied = q4 ^ (i * 0x11111111);
        int2 result = neg_2perm(q_varied);
        acc_x += result.x;
        acc_y += result.y;
    }

    out[idx] = acc_x + acc_y;
}

// ============================================================================
// Even more compute: simulate dot product accumulation
// ============================================================================

__global__ void bench_baseline_dotprod(const uint32_t* __restrict__ qs,
                                        const int8_t* __restrict__ activations,
                                        int* __restrict__ out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    uint32_t q4 = qs[idx];
    int2 weights = baseline_6perm(q4);

    // Simulate dot product with activations
    int8_t* w = (int8_t*)&weights;
    int8_t a[8];
    *((int2*)a) = *((int2*)&activations[idx * 8]);

    int sum = 0;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        sum += (int)w[i] * (int)a[i];
    }

    out[idx] = sum;
}

__global__ void bench_neg2perm_dotprod(const uint32_t* __restrict__ qs,
                                        const int8_t* __restrict__ activations,
                                        int* __restrict__ out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    uint32_t q4 = qs[idx];
    int2 weights = neg_2perm(q4);

    int8_t* w = (int8_t*)&weights;
    int8_t a[8];
    *((int2*)a) = *((int2*)&activations[idx * 8]);

    int sum = 0;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        sum += (int)w[i] * (int)a[i];
    }

    out[idx] = sum;
}

int main() {
    printf("Compute-Bound MXFP4 Benchmark\n");
    printf("==============================\n\n");

    const int N = 1024 * 1024;

    uint32_t* d_data;
    int* d_out;
    int8_t* d_act;

    hipMalloc(&d_data, N * sizeof(uint32_t));
    hipMalloc(&d_out, N * sizeof(int));
    hipMalloc(&d_act, N * 8 * sizeof(int8_t));

    // All-negative test data
    uint32_t* h_data = new uint32_t[N];
    for (int i = 0; i < N; i++) {
        h_data[i] = ((i * 2654435761u) & 0x77777777) | 0x88888888;  // All negative
    }
    hipMemcpy(d_data, h_data, N * sizeof(uint32_t), hipMemcpyHostToDevice);

    // Random activations
    int8_t* h_act = new int8_t[N * 8];
    for (int i = 0; i < N * 8; i++) h_act[i] = (i * 17) & 0x7F;
    hipMemcpy(d_act, h_act, N * 8, hipMemcpyHostToDevice);

    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);

    const int ITERS = 100;
    float ms_base, ms_opt;

    // Warmup
    bench_baseline_compute<8><<<N/256, 256>>>(d_data, d_out, N);
    hipDeviceSynchronize();

    printf("Test 1: Multiple lookups per thread (compute-bound)\n");
    printf("---------------------------------------------------\n");

    // Test with different lookup counts
    for (int lookups : {4, 8, 16, 32}) {
        printf("\n%d lookups per thread:\n", lookups);

        if (lookups == 4) {
            hipEventRecord(start);
            for (int i = 0; i < ITERS; i++)
                bench_baseline_compute<4><<<N/256, 256>>>(d_data, d_out, N);
            hipEventRecord(stop);
            hipEventSynchronize(stop);
            hipEventElapsedTime(&ms_base, start, stop);

            hipEventRecord(start);
            for (int i = 0; i < ITERS; i++)
                bench_neg2perm_compute<4><<<N/256, 256>>>(d_data, d_out, N);
            hipEventRecord(stop);
            hipEventSynchronize(stop);
            hipEventElapsedTime(&ms_opt, start, stop);
        } else if (lookups == 8) {
            hipEventRecord(start);
            for (int i = 0; i < ITERS; i++)
                bench_baseline_compute<8><<<N/256, 256>>>(d_data, d_out, N);
            hipEventRecord(stop);
            hipEventSynchronize(stop);
            hipEventElapsedTime(&ms_base, start, stop);

            hipEventRecord(start);
            for (int i = 0; i < ITERS; i++)
                bench_neg2perm_compute<8><<<N/256, 256>>>(d_data, d_out, N);
            hipEventRecord(stop);
            hipEventSynchronize(stop);
            hipEventElapsedTime(&ms_opt, start, stop);
        } else if (lookups == 16) {
            hipEventRecord(start);
            for (int i = 0; i < ITERS; i++)
                bench_baseline_compute<16><<<N/256, 256>>>(d_data, d_out, N);
            hipEventRecord(stop);
            hipEventSynchronize(stop);
            hipEventElapsedTime(&ms_base, start, stop);

            hipEventRecord(start);
            for (int i = 0; i < ITERS; i++)
                bench_neg2perm_compute<16><<<N/256, 256>>>(d_data, d_out, N);
            hipEventRecord(stop);
            hipEventSynchronize(stop);
            hipEventElapsedTime(&ms_opt, start, stop);
        } else {
            hipEventRecord(start);
            for (int i = 0; i < ITERS; i++)
                bench_baseline_compute<32><<<N/256, 256>>>(d_data, d_out, N);
            hipEventRecord(stop);
            hipEventSynchronize(stop);
            hipEventElapsedTime(&ms_base, start, stop);

            hipEventRecord(start);
            for (int i = 0; i < ITERS; i++)
                bench_neg2perm_compute<32><<<N/256, 256>>>(d_data, d_out, N);
            hipEventRecord(stop);
            hipEventSynchronize(stop);
            hipEventElapsedTime(&ms_opt, start, stop);
        }

        printf("  Baseline (6-perm): %.3f ms\n", ms_base);
        printf("  Optimized (2-perm): %.3f ms (%.1f%% %s)\n",
               ms_opt, 100.0*fabs(ms_opt-ms_base)/ms_base,
               ms_opt < ms_base ? "FASTER" : "slower");
    }

    printf("\n\nTest 2: Dot product simulation (realistic MMQ pattern)\n");
    printf("------------------------------------------------------\n");

    hipEventRecord(start);
    for (int i = 0; i < ITERS; i++)
        bench_baseline_dotprod<<<N/256, 256>>>(d_data, d_act, d_out, N);
    hipEventRecord(stop);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&ms_base, start, stop);

    hipEventRecord(start);
    for (int i = 0; i < ITERS; i++)
        bench_neg2perm_dotprod<<<N/256, 256>>>(d_data, d_act, d_out, N);
    hipEventRecord(stop);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&ms_opt, start, stop);

    printf("  Baseline (6-perm): %.3f ms\n", ms_base);
    printf("  Optimized (2-perm): %.3f ms (%.1f%% %s)\n",
           ms_opt, 100.0*fabs(ms_opt-ms_base)/ms_base,
           ms_opt < ms_base ? "FASTER" : "slower");

    printf("\n\nAnalysis:\n");
    printf("=========\n");
    printf("- Previous benchmark was memory-bound (load → compute → store)\n");
    printf("- Real MMQ kernels are compute-bound (multiple ops per load)\n");
    printf("- The 2-perm optimization saves 4 perms per lookup\n");
    printf("- With 6 perms baseline and 2 perms optimized = 66%% perm reduction\n");
    printf("- Actual speedup depends on other ALU ops in the kernel\n");

    delete[] h_data;
    delete[] h_act;
    hipFree(d_data);
    hipFree(d_out);
    hipFree(d_act);
    hipEventDestroy(start);
    hipEventDestroy(stop);

    return 0;
}
