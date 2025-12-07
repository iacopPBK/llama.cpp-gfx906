// Fixed BFI-based selection for MXFP4 lookup
// Goal: 4 v_perm (lookups) + 2 v_bfi (select) instead of 6 v_perm

#include <hip/hip_runtime.h>
#include <stdio.h>
#include <chrono>

// Original: 6 v_perm
__device__ __forceinline__ int2 lookup_6perm(const int q4, const uint32_t* values) {
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

// Fixed: 4 v_perm + BFI with correct mask expansion
// Uses x |= x<<1; x |= x<<2; x |= x<<4 to expand each bit to full byte
__device__ __forceinline__ int2 lookup_4perm_bfi_fixed(const int q4, const uint32_t* values) {
    const uint32_t q_even = q4;
    const uint32_t q_odd  = (q4 >> 4);

    // 4 lookups
    uint32_t v_even_low = __builtin_amdgcn_perm(values[1], values[0], q_even & 0x07070707);
    uint32_t v_odd_low = __builtin_amdgcn_perm(values[1], values[0], q_odd & 0x07070707);
    uint32_t v_even_high = __builtin_amdgcn_perm(values[3], values[2], q_even & 0x07070707);
    uint32_t v_odd_high = __builtin_amdgcn_perm(values[3], values[2], q_odd & 0x07070707);

    // Correct mask expansion: 0x01 -> 0xFF per byte
    uint32_t b3e = (q_even >> 3) & 0x01010101;
    uint32_t me = b3e;
    me |= me << 1;  // 0x01 -> 0x03
    me |= me << 2;  // 0x03 -> 0x0F
    me |= me << 4;  // 0x0F -> 0xFF

    uint32_t b3o = (q_odd >> 3) & 0x01010101;
    uint32_t mo = b3o;
    mo |= mo << 1;
    mo |= mo << 2;
    mo |= mo << 4;

    // BFI selection: (high & mask) | (low & ~mask)
    uint32_t res_x = (v_even_high & me) | (v_even_low & ~me);
    uint32_t res_y = (v_odd_high & mo) | (v_odd_low & ~mo);

    return make_int2(res_x, res_y);
}

// Test kernels
__global__ void test_6perm(const int* q4_data, int2* output, const int8_t* table, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    const uint32_t* values = (const uint32_t*)table;
    output[idx] = lookup_6perm(q4_data[idx], values);
}

__global__ void test_4perm_bfi_fixed(const int* q4_data, int2* output, const int8_t* table, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    const uint32_t* values = (const uint32_t*)table;
    output[idx] = lookup_4perm_bfi_fixed(q4_data[idx], values);
}

int main() {
    const int N = 1024 * 1024 * 16;
    const int iterations = 100;

    // Use actual MXFP4 table: {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12}
    int8_t h_table[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};

    int* d_q4;
    int2* d_output;
    int8_t* d_table;

    hipMalloc(&d_q4, N * sizeof(int));
    hipMalloc(&d_output, N * sizeof(int2));
    hipMalloc(&d_table, 16);
    hipMemcpy(d_table, h_table, 16, hipMemcpyHostToDevice);

    // Generate test data with full range of 4-bit indices
    int* h_q4 = new int[N];
    for (int i = 0; i < N; i++) {
        h_q4[i] = rand();  // Full random - each nibble 0-15
    }
    hipMemcpy(d_q4, h_q4, N * sizeof(int), hipMemcpyHostToDevice);

    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);

    // Warmup
    test_6perm<<<grid, block>>>(d_q4, d_output, d_table, N);
    hipDeviceSynchronize();

    // Verify correctness
    int2* h_ref = new int2[N];
    int2* h_test = new int2[N];

    test_6perm<<<grid, block>>>(d_q4, d_output, d_table, N);
    hipMemcpy(h_ref, d_output, N * sizeof(int2), hipMemcpyDeviceToHost);

    test_4perm_bfi_fixed<<<grid, block>>>(d_q4, d_output, d_table, N);
    hipMemcpy(h_test, d_output, N * sizeof(int2), hipMemcpyDeviceToHost);

    int errors = 0;
    for (int i = 0; i < N && errors < 10; i++) {
        if (h_ref[i].x != h_test[i].x || h_ref[i].y != h_test[i].y) {
            printf("Mismatch at %d: ref=(%08x,%08x) test=(%08x,%08x) q4=%08x\n",
                   i, h_ref[i].x, h_ref[i].y, h_test[i].x, h_test[i].y, h_q4[i]);
            errors++;
        }
    }
    printf("4perm_bfi_fixed: %s (%d errors)\n", errors == 0 ? "PASS" : "FAIL", errors);

    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        test_6perm<<<grid, block>>>(d_q4, d_output, d_table, N);
    }
    hipDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    double t_6perm = std::chrono::duration<double, std::milli>(end - start).count();

    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        test_4perm_bfi_fixed<<<grid, block>>>(d_q4, d_output, d_table, N);
    }
    hipDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    double t_bfi = std::chrono::duration<double, std::milli>(end - start).count();

    printf("\nBenchmark (%d iterations, %d lookups):\n", iterations, N);
    printf("  6 v_perm (baseline): %.2f ms\n", t_6perm);
    printf("  4 v_perm + BFI fix:  %.2f ms (%+.1f%%)\n", t_bfi, (t_bfi/t_6perm - 1)*100);

    hipFree(d_q4);
    hipFree(d_output);
    hipFree(d_table);
    delete[] h_q4;
    delete[] h_ref;
    delete[] h_test;

    return 0;
}
