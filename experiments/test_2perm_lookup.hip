// 2 v_perm lookup using MXFP4 symmetry with proper per-byte negation
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <chrono>

// MXFP4: {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12}
// Key insight: table[i+8] = -table[i], and val=0 only when idx=0 or 8

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

// 2 v_perm with carry-safe negation
// Key: val=0 only when idx=0, so we can mask out the +1 for those bytes
__device__ __forceinline__ int2 lookup_2perm_fixed(const int q4, const uint32_t* half_values) {
    const uint32_t q_even = q4;
    const uint32_t q_odd  = (q4 >> 4);

    // Extract 3-bit indices (0-7)
    const uint32_t idx_even = q_even & 0x07070707;
    const uint32_t idx_odd  = q_odd & 0x07070707;

    // Extract sign bits (bit 3)
    const uint32_t sign_even = (q_even >> 3) & 0x01010101;
    const uint32_t sign_odd  = (q_odd >> 3) & 0x01010101;

    // Only 2 v_perm lookups!
    uint32_t val_even = __builtin_amdgcn_perm(half_values[1], half_values[0], idx_even);
    uint32_t val_odd  = __builtin_amdgcn_perm(half_values[1], half_values[0], idx_odd);

    // Generate sign masks (0x00 or 0xFF per byte)
    uint32_t sm_even = sign_even; sm_even |= sm_even<<1; sm_even |= sm_even<<2; sm_even |= sm_even<<4;
    uint32_t sm_odd  = sign_odd;  sm_odd  |= sm_odd<<1;  sm_odd  |= sm_odd<<2;  sm_odd  |= sm_odd<<4;

    // Detect non-zero indices (idx != 0 means val != 0)
    // If any bit in idx is set, the byte is non-zero
    uint32_t nz_even = idx_even; nz_even |= nz_even>>1; nz_even |= nz_even>>2;
    nz_even |= nz_even<<1; nz_even |= nz_even<<2; nz_even |= nz_even<<4;  // Expand to 0xFF per non-zero byte

    uint32_t nz_odd = idx_odd; nz_odd |= nz_odd>>1; nz_odd |= nz_odd>>2;
    nz_odd |= nz_odd<<1; nz_odd |= nz_odd<<2; nz_odd |= nz_odd<<4;

    // For negation: ~val + 1, but only add 1 if val != 0 (to avoid carry corruption)
    // When val=0: ~0 = 0xFF, and we DON'T add 1, so result stays 0xFF... wrong!
    // We want result=0 when val=0, regardless of sign

    // Better approach: XOR with sign_mask, then add (sign & non_zero)
    uint32_t flipped_even = val_even ^ sm_even;  // ~val where sign=1
    uint32_t flipped_odd  = val_odd  ^ sm_odd;

    // Add 1 only where sign=1 AND val!=0
    uint32_t to_add_even = sign_even & (nz_even & 0x01010101);
    uint32_t to_add_odd  = sign_odd  & (nz_odd  & 0x01010101);

    // Now add - but we still have carry issues!
    // The carry only corrupts if a byte overflows, which happens when byte=0xFF and we add 1
    // After XOR: if val=0 and sign=1, flipped=0xFF; but we don't add 1 (to_add=0)
    // If val!=0 and sign=1, flipped=~val, and we add 1 -> proper negation, no overflow for val<=12

    uint32_t res_even = flipped_even + to_add_even;
    uint32_t res_odd  = flipped_odd  + to_add_odd;

    // BUT: we still need to handle the case where val=0 and sign=1
    // In that case, flipped=0xFF and we want result=0
    // Currently we get 0xFF (since to_add=0)

    // Fix: mask out bytes where val=0 and sign=1
    // These should be 0, not 0xFF
    uint32_t zero_and_sign_even = sm_even & ~nz_even;  // 0xFF where val=0 AND sign=1
    uint32_t zero_and_sign_odd  = sm_odd  & ~nz_odd;

    res_even &= ~zero_and_sign_even;  // Clear bytes where val=0 and sign=1
    res_odd  &= ~zero_and_sign_odd;

    return make_int2(res_even, res_odd);
}

// Alternative: use byte-wise operations (may be slower but cleaner)
__device__ __forceinline__ int2 lookup_2perm_bytewise(const int q4, const uint32_t* half_values) {
    const uint32_t q_even = q4;
    const uint32_t q_odd  = (q4 >> 4);

    const uint32_t idx_even = q_even & 0x07070707;
    const uint32_t idx_odd  = q_odd & 0x07070707;
    const uint32_t sign_even = (q_even >> 3) & 0x01010101;
    const uint32_t sign_odd  = (q_odd >> 3) & 0x01010101;

    // 2 v_perm lookups
    uint32_t val_even = __builtin_amdgcn_perm(half_values[1], half_values[0], idx_even);
    uint32_t val_odd  = __builtin_amdgcn_perm(half_values[1], half_values[0], idx_odd);

    // Per-byte negation using extraction and recombination
    uint32_t res_even = 0, res_odd = 0;

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int shift = i * 8;
        int8_t byte_val = (val_even >> shift) & 0xFF;
        int8_t sign_bit = (sign_even >> shift) & 0x01;
        int8_t result = sign_bit ? -byte_val : byte_val;
        res_even |= ((uint32_t)(uint8_t)result) << shift;

        byte_val = (val_odd >> shift) & 0xFF;
        sign_bit = (sign_odd >> shift) & 0x01;
        result = sign_bit ? -byte_val : byte_val;
        res_odd |= ((uint32_t)(uint8_t)result) << shift;
    }

    return make_int2(res_even, res_odd);
}

// Test kernels
__global__ void test_baseline_k(const int* q4_data, int2* output, const int8_t* table, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    output[idx] = lookup_baseline(q4_data[idx], (const uint32_t*)table);
}

__global__ void test_2perm_fixed_k(const int* q4_data, int2* output, const int8_t* half_table, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    output[idx] = lookup_2perm_fixed(q4_data[idx], (const uint32_t*)half_table);
}

__global__ void test_2perm_bytewise_k(const int* q4_data, int2* output, const int8_t* half_table, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    output[idx] = lookup_2perm_bytewise(q4_data[idx], (const uint32_t*)half_table);
}

int main() {
    const int N = 1024 * 1024 * 16;
    const int iterations = 100;

    int8_t h_table[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
    int8_t h_half[8] = {0, 1, 2, 3, 4, 6, 8, 12};

    int* d_q4;
    int2* d_output;
    int8_t *d_table, *d_half;

    hipMalloc(&d_q4, N * sizeof(int));
    hipMalloc(&d_output, N * sizeof(int2));
    hipMalloc(&d_table, 16);
    hipMalloc(&d_half, 8);
    hipMemcpy(d_table, h_table, 16, hipMemcpyHostToDevice);
    hipMemcpy(d_half, h_half, 8, hipMemcpyHostToDevice);

    int* h_q4 = new int[N];
    for (int i = 0; i < N; i++) h_q4[i] = rand();
    hipMemcpy(d_q4, h_q4, N * sizeof(int), hipMemcpyHostToDevice);

    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);

    // Get reference
    int2* h_ref = new int2[N];
    int2* h_test = new int2[N];

    test_baseline_k<<<grid, block>>>(d_q4, d_output, d_table, N);
    hipDeviceSynchronize();
    hipMemcpy(h_ref, d_output, N * sizeof(int2), hipMemcpyDeviceToHost);

    // Test 2perm_fixed
    test_2perm_fixed_k<<<grid, block>>>(d_q4, d_output, d_half, N);
    hipMemcpy(h_test, d_output, N * sizeof(int2), hipMemcpyDeviceToHost);
    int errors = 0;
    for (int i = 0; i < N && errors < 10; i++) {
        if (h_ref[i].x != h_test[i].x || h_ref[i].y != h_test[i].y) {
            if (errors < 3) printf("2perm_fixed err %d: ref(%08x,%08x) got(%08x,%08x) q4=%08x\n",
                   i, h_ref[i].x, h_ref[i].y, h_test[i].x, h_test[i].y, h_q4[i]);
            errors++;
        }
    }
    printf("2perm_fixed: %s (%d errors)\n", errors == 0 ? "PASS" : "FAIL", errors);

    // Test 2perm_bytewise
    test_2perm_bytewise_k<<<grid, block>>>(d_q4, d_output, d_half, N);
    hipMemcpy(h_test, d_output, N * sizeof(int2), hipMemcpyDeviceToHost);
    errors = 0;
    for (int i = 0; i < N && errors < 10; i++) {
        if (h_ref[i].x != h_test[i].x || h_ref[i].y != h_test[i].y) {
            if (errors < 3) printf("2perm_bytewise err %d: ref(%08x,%08x) got(%08x,%08x) q4=%08x\n",
                   i, h_ref[i].x, h_ref[i].y, h_test[i].x, h_test[i].y, h_q4[i]);
            errors++;
        }
    }
    printf("2perm_bytewise: %s (%d errors)\n", errors == 0 ? "PASS" : "FAIL", errors);

    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        test_baseline_k<<<grid, block>>>(d_q4, d_output, d_table, N);
    }
    hipDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    double t_base = std::chrono::duration<double, std::milli>(end - start).count();

    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        test_2perm_fixed_k<<<grid, block>>>(d_q4, d_output, d_half, N);
    }
    hipDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    double t_fixed = std::chrono::duration<double, std::milli>(end - start).count();

    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        test_2perm_bytewise_k<<<grid, block>>>(d_q4, d_output, d_half, N);
    }
    hipDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    double t_bytewise = std::chrono::duration<double, std::milli>(end - start).count();

    printf("\nBenchmark (%d iters, %dM lookups):\n", iterations, N/1000000);
    printf("  Baseline (6 perm):     %.2f ms\n", t_base);
    printf("  2perm_fixed:           %.2f ms (%+.1f%%)\n", t_fixed, (t_fixed/t_base - 1)*100);
    printf("  2perm_bytewise:        %.2f ms (%+.1f%%)\n", t_bytewise, (t_bytewise/t_base - 1)*100);

    hipFree(d_q4);
    hipFree(d_output);
    hipFree(d_table);
    hipFree(d_half);
    delete[] h_q4;
    delete[] h_ref;
    delete[] h_test;

    return 0;
}
