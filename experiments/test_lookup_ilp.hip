// ILP and multi-value optimizations for MXFP4 lookup
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <chrono>

// Baseline: 6 v_perm
__device__ __forceinline__ int2 lookup_baseline(const int q4, const uint32_t* values) {
    const uint32_t q_even = q4;
    const uint32_t q_odd  = (q4 >> 4);

    uint32_t v_even_low = __builtin_amdgcn_perm(values[1], values[0], q_even & 0x07070707);
    uint32_t v_odd_low = __builtin_amdgcn_perm(values[1], values[0], q_odd & 0x07070707);
    uint32_t v_even_high = __builtin_amdgcn_perm(values[3], values[2], q_even & 0x07070707);
    uint32_t v_odd_high = __builtin_amdgcn_perm(values[3], values[2], q_odd & 0x07070707);

    uint32_t mask_even = 0x03020100 | ((q_even & 0x08080808) >> 1);
    uint32_t res_x = __builtin_amdgcn_perm(v_even_high, v_even_low, mask_even);
    uint32_t mask_odd = 0x03020100 | ((q_odd & 0x08080808) >> 1);
    uint32_t res_y = __builtin_amdgcn_perm(v_odd_high, v_odd_low, mask_odd);

    return make_int2(res_x, res_y);
}

// ILP optimized: interleave operations to hide latency
__device__ __forceinline__ int2 lookup_ilp(const int q4, const uint32_t* values) {
    const uint32_t q_even = q4;
    const uint32_t q_odd  = (q4 >> 4);

    // Extract selectors early (can execute while loads settle)
    const uint32_t sel_even = q_even & 0x07070707;
    const uint32_t sel_odd  = q_odd & 0x07070707;

    // Start mask computation early (independent of lookups)
    uint32_t b3e = (q_even >> 3) & 0x01010101;
    uint32_t b3o = (q_odd >> 3) & 0x01010101;

    // 4 lookups - interleave even/odd to maximize ILP
    uint32_t v_even_low = __builtin_amdgcn_perm(values[1], values[0], sel_even);
    uint32_t v_odd_low = __builtin_amdgcn_perm(values[1], values[0], sel_odd);

    // Continue mask expansion while lookups execute
    uint32_t me = b3e; me |= me << 1; me |= me << 2;
    uint32_t mo = b3o; mo |= mo << 1; mo |= mo << 2;

    uint32_t v_even_high = __builtin_amdgcn_perm(values[3], values[2], sel_even);
    uint32_t v_odd_high = __builtin_amdgcn_perm(values[3], values[2], sel_odd);

    // Finish mask expansion
    me |= me << 4;
    mo |= mo << 4;

    // BFI selection
    uint32_t res_x = (v_even_high & me) | (v_even_low & ~me);
    uint32_t res_y = (v_odd_high & mo) | (v_odd_low & ~mo);

    return make_int2(res_x, res_y);
}

// Process 2 values at once for better throughput
__device__ __forceinline__ void lookup_2x(
    const int q4_a, const int q4_b, const uint32_t* values,
    int2& out_a, int2& out_b) {

    const uint32_t q_even_a = q4_a;
    const uint32_t q_odd_a  = (q4_a >> 4);
    const uint32_t q_even_b = q4_b;
    const uint32_t q_odd_b  = (q4_b >> 4);

    const uint32_t sel_even_a = q_even_a & 0x07070707;
    const uint32_t sel_odd_a  = q_odd_a & 0x07070707;
    const uint32_t sel_even_b = q_even_b & 0x07070707;
    const uint32_t sel_odd_b  = q_odd_b & 0x07070707;

    // Start all mask computations
    uint32_t b3e_a = (q_even_a >> 3) & 0x01010101;
    uint32_t b3o_a = (q_odd_a >> 3) & 0x01010101;
    uint32_t b3e_b = (q_even_b >> 3) & 0x01010101;
    uint32_t b3o_b = (q_odd_b >> 3) & 0x01010101;

    // Interleaved lookups for both values
    uint32_t v_even_low_a = __builtin_amdgcn_perm(values[1], values[0], sel_even_a);
    uint32_t v_even_low_b = __builtin_amdgcn_perm(values[1], values[0], sel_even_b);
    uint32_t v_odd_low_a = __builtin_amdgcn_perm(values[1], values[0], sel_odd_a);
    uint32_t v_odd_low_b = __builtin_amdgcn_perm(values[1], values[0], sel_odd_b);

    // Mask expansion for A
    uint32_t me_a = b3e_a; me_a |= me_a << 1; me_a |= me_a << 2; me_a |= me_a << 4;
    uint32_t mo_a = b3o_a; mo_a |= mo_a << 1; mo_a |= mo_a << 2; mo_a |= mo_a << 4;

    uint32_t v_even_high_a = __builtin_amdgcn_perm(values[3], values[2], sel_even_a);
    uint32_t v_even_high_b = __builtin_amdgcn_perm(values[3], values[2], sel_even_b);
    uint32_t v_odd_high_a = __builtin_amdgcn_perm(values[3], values[2], sel_odd_a);
    uint32_t v_odd_high_b = __builtin_amdgcn_perm(values[3], values[2], sel_odd_b);

    // Mask expansion for B
    uint32_t me_b = b3e_b; me_b |= me_b << 1; me_b |= me_b << 2; me_b |= me_b << 4;
    uint32_t mo_b = b3o_b; mo_b |= mo_b << 1; mo_b |= mo_b << 2; mo_b |= mo_b << 4;

    // BFI selections
    out_a.x = (v_even_high_a & me_a) | (v_even_low_a & ~me_a);
    out_a.y = (v_odd_high_a & mo_a) | (v_odd_low_a & ~mo_a);
    out_b.x = (v_even_high_b & me_b) | (v_even_low_b & ~me_b);
    out_b.y = (v_odd_high_b & mo_b) | (v_odd_low_b & ~mo_b);
}

// Process 4 values at once
__device__ __forceinline__ void lookup_4x(
    const int4 q4_vec, const uint32_t* values,
    int2& out_0, int2& out_1, int2& out_2, int2& out_3) {

    // Extract all even/odd pairs
    const uint32_t q_even_0 = q4_vec.x, q_odd_0 = q4_vec.x >> 4;
    const uint32_t q_even_1 = q4_vec.y, q_odd_1 = q4_vec.y >> 4;
    const uint32_t q_even_2 = q4_vec.z, q_odd_2 = q4_vec.z >> 4;
    const uint32_t q_even_3 = q4_vec.w, q_odd_3 = q4_vec.w >> 4;

    // All selectors
    const uint32_t mask07 = 0x07070707;
    const uint32_t sel_e0 = q_even_0 & mask07, sel_o0 = q_odd_0 & mask07;
    const uint32_t sel_e1 = q_even_1 & mask07, sel_o1 = q_odd_1 & mask07;
    const uint32_t sel_e2 = q_even_2 & mask07, sel_o2 = q_odd_2 & mask07;
    const uint32_t sel_e3 = q_even_3 & mask07, sel_o3 = q_odd_3 & mask07;

    // All bit3 extractions
    const uint32_t mask01 = 0x01010101;
    uint32_t b3e0 = (q_even_0 >> 3) & mask01, b3o0 = (q_odd_0 >> 3) & mask01;
    uint32_t b3e1 = (q_even_1 >> 3) & mask01, b3o1 = (q_odd_1 >> 3) & mask01;
    uint32_t b3e2 = (q_even_2 >> 3) & mask01, b3o2 = (q_odd_2 >> 3) & mask01;
    uint32_t b3e3 = (q_even_3 >> 3) & mask01, b3o3 = (q_odd_3 >> 3) & mask01;

    // 16 lookups - all low table
    uint32_t vel0 = __builtin_amdgcn_perm(values[1], values[0], sel_e0);
    uint32_t vol0 = __builtin_amdgcn_perm(values[1], values[0], sel_o0);
    uint32_t vel1 = __builtin_amdgcn_perm(values[1], values[0], sel_e1);
    uint32_t vol1 = __builtin_amdgcn_perm(values[1], values[0], sel_o1);
    uint32_t vel2 = __builtin_amdgcn_perm(values[1], values[0], sel_e2);
    uint32_t vol2 = __builtin_amdgcn_perm(values[1], values[0], sel_o2);
    uint32_t vel3 = __builtin_amdgcn_perm(values[1], values[0], sel_e3);
    uint32_t vol3 = __builtin_amdgcn_perm(values[1], values[0], sel_o3);

    // Expand masks while lookups execute
    uint32_t me0 = b3e0; me0 |= me0<<1; me0 |= me0<<2; me0 |= me0<<4;
    uint32_t mo0 = b3o0; mo0 |= mo0<<1; mo0 |= mo0<<2; mo0 |= mo0<<4;
    uint32_t me1 = b3e1; me1 |= me1<<1; me1 |= me1<<2; me1 |= me1<<4;
    uint32_t mo1 = b3o1; mo1 |= mo1<<1; mo1 |= mo1<<2; mo1 |= mo1<<4;

    // All high table lookups
    uint32_t veh0 = __builtin_amdgcn_perm(values[3], values[2], sel_e0);
    uint32_t voh0 = __builtin_amdgcn_perm(values[3], values[2], sel_o0);
    uint32_t veh1 = __builtin_amdgcn_perm(values[3], values[2], sel_e1);
    uint32_t voh1 = __builtin_amdgcn_perm(values[3], values[2], sel_o1);
    uint32_t veh2 = __builtin_amdgcn_perm(values[3], values[2], sel_e2);
    uint32_t voh2 = __builtin_amdgcn_perm(values[3], values[2], sel_o2);
    uint32_t veh3 = __builtin_amdgcn_perm(values[3], values[2], sel_e3);
    uint32_t voh3 = __builtin_amdgcn_perm(values[3], values[2], sel_o3);

    // Remaining mask expansions
    uint32_t me2 = b3e2; me2 |= me2<<1; me2 |= me2<<2; me2 |= me2<<4;
    uint32_t mo2 = b3o2; mo2 |= mo2<<1; mo2 |= mo2<<2; mo2 |= mo2<<4;
    uint32_t me3 = b3e3; me3 |= me3<<1; me3 |= me3<<2; me3 |= me3<<4;
    uint32_t mo3 = b3o3; mo3 |= mo3<<1; mo3 |= mo3<<2; mo3 |= mo3<<4;

    // All BFI selections
    out_0.x = (veh0 & me0) | (vel0 & ~me0);
    out_0.y = (voh0 & mo0) | (vol0 & ~mo0);
    out_1.x = (veh1 & me1) | (vel1 & ~me1);
    out_1.y = (voh1 & mo1) | (vol1 & ~mo1);
    out_2.x = (veh2 & me2) | (vel2 & ~me2);
    out_2.y = (voh2 & mo2) | (vol2 & ~mo2);
    out_3.x = (veh3 & me3) | (vel3 & ~me3);
    out_3.y = (voh3 & mo3) | (vol3 & ~mo3);
}

// Test kernels
__global__ void test_baseline(const int* q4_data, int2* output, const int8_t* table, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    const uint32_t* values = (const uint32_t*)table;
    output[idx] = lookup_baseline(q4_data[idx], values);
}

__global__ void test_ilp(const int* q4_data, int2* output, const int8_t* table, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    const uint32_t* values = (const uint32_t*)table;
    output[idx] = lookup_ilp(q4_data[idx], values);
}

__global__ void test_2x(const int* q4_data, int2* output, const int8_t* table, int n) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    if (idx + 1 >= n) return;
    const uint32_t* values = (const uint32_t*)table;
    int2 out_a, out_b;
    lookup_2x(q4_data[idx], q4_data[idx+1], values, out_a, out_b);
    output[idx] = out_a;
    output[idx+1] = out_b;
}

__global__ void test_4x(const int* q4_data, int2* output, const int8_t* table, int n) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (idx + 3 >= n) return;
    const uint32_t* values = (const uint32_t*)table;
    int4 q4_vec = make_int4(q4_data[idx], q4_data[idx+1], q4_data[idx+2], q4_data[idx+3]);
    int2 out_0, out_1, out_2, out_3;
    lookup_4x(q4_vec, values, out_0, out_1, out_2, out_3);
    output[idx] = out_0;
    output[idx+1] = out_1;
    output[idx+2] = out_2;
    output[idx+3] = out_3;
}

// 4x with vectorized load/store
__global__ void test_4x_vec(const int4* q4_data, int2* output, const int8_t* table, int n) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (idx + 3 >= n) return;
    const uint32_t* values = (const uint32_t*)table;

    // Single 128-bit load
    int4 q4_vec = q4_data[blockIdx.x * blockDim.x + threadIdx.x];
    int2 out_0, out_1, out_2, out_3;
    lookup_4x(q4_vec, values, out_0, out_1, out_2, out_3);

    // Vectorized stores
    ((int4*)&output[idx])[0] = make_int4(out_0.x, out_0.y, out_1.x, out_1.y);
    ((int4*)&output[idx+2])[0] = make_int4(out_2.x, out_2.y, out_3.x, out_3.y);
}

int main() {
    const int N = 1024 * 1024 * 16;
    const int iterations = 100;

    int8_t h_table[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};

    int* d_q4;
    int2* d_output;
    int8_t* d_table;

    hipMalloc(&d_q4, N * sizeof(int));
    hipMalloc(&d_output, N * sizeof(int2));
    hipMalloc(&d_table, 16);
    hipMemcpy(d_table, h_table, 16, hipMemcpyHostToDevice);

    int* h_q4 = new int[N];
    for (int i = 0; i < N; i++) h_q4[i] = rand();
    hipMemcpy(d_q4, h_q4, N * sizeof(int), hipMemcpyHostToDevice);

    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);
    dim3 grid_2x((N/2 + block.x - 1) / block.x);
    dim3 grid_4x((N/4 + block.x - 1) / block.x);

    // Warmup & verify
    test_baseline<<<grid, block>>>(d_q4, d_output, d_table, N);
    hipDeviceSynchronize();

    int2* h_ref = new int2[N];
    int2* h_test = new int2[N];

    test_baseline<<<grid, block>>>(d_q4, d_output, d_table, N);
    hipMemcpy(h_ref, d_output, N * sizeof(int2), hipMemcpyDeviceToHost);

    // Verify ILP
    test_ilp<<<grid, block>>>(d_q4, d_output, d_table, N);
    hipMemcpy(h_test, d_output, N * sizeof(int2), hipMemcpyDeviceToHost);
    int errors = 0;
    for (int i = 0; i < N && errors < 5; i++) {
        if (h_ref[i].x != h_test[i].x || h_ref[i].y != h_test[i].y) errors++;
    }
    printf("ILP: %s\n", errors == 0 ? "PASS" : "FAIL");

    // Verify 2x
    test_2x<<<grid_2x, block>>>(d_q4, d_output, d_table, N);
    hipMemcpy(h_test, d_output, N * sizeof(int2), hipMemcpyDeviceToHost);
    errors = 0;
    for (int i = 0; i < N && errors < 5; i++) {
        if (h_ref[i].x != h_test[i].x || h_ref[i].y != h_test[i].y) errors++;
    }
    printf("2x: %s\n", errors == 0 ? "PASS" : "FAIL");

    // Verify 4x
    test_4x<<<grid_4x, block>>>(d_q4, d_output, d_table, N);
    hipMemcpy(h_test, d_output, N * sizeof(int2), hipMemcpyDeviceToHost);
    errors = 0;
    for (int i = 0; i < N && errors < 5; i++) {
        if (h_ref[i].x != h_test[i].x || h_ref[i].y != h_test[i].y) errors++;
    }
    printf("4x: %s\n", errors == 0 ? "PASS" : "FAIL");

    // Benchmark
    auto bench = [&](auto kernel, dim3 g, const char* name) {
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; i++) {
            kernel<<<g, block>>>(d_q4, d_output, d_table, N);
        }
        hipDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start).count();
    };

    double t_base = bench(test_baseline, grid, "baseline");
    double t_ilp = bench(test_ilp, grid, "ilp");
    double t_2x = bench(test_2x, grid_2x, "2x");
    double t_4x = bench(test_4x, grid_4x, "4x");

    printf("\nBenchmark (%d iters, %dM lookups):\n", iterations, N/1000000);
    printf("  Baseline (6 perm):  %.2f ms\n", t_base);
    printf("  ILP optimized:      %.2f ms (%+.1f%%)\n", t_ilp, (t_ilp/t_base - 1)*100);
    printf("  2x per thread:      %.2f ms (%+.1f%%)\n", t_2x, (t_2x/t_base - 1)*100);
    printf("  4x per thread:      %.2f ms (%+.1f%%)\n", t_4x, (t_4x/t_base - 1)*100);

    hipFree(d_q4);
    hipFree(d_output);
    hipFree(d_table);
    delete[] h_q4;
    delete[] h_ref;
    delete[] h_test;

    return 0;
}
