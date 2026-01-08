// 8-entry table + sign flip for MXFP4
// Exploits: table[i+8] = -table[i] symmetry
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <chrono>

// MXFP4 table: {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12}
// Half table: {0, 1, 2, 3, 4, 6, 8, 12}

// Baseline: 6 v_perm (current)
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

// 8-entry approach: 2 v_perm + sign handling
// Problem: ~x + 1 causes carry between bytes when x=0
// Solution: Use XOR-based conditional negation that handles 0 correctly
__device__ __forceinline__ int2 lookup_8entry_v1(const int q4, const uint32_t* half_values) {
    const uint32_t q_even = q4;
    const uint32_t q_odd  = (q4 >> 4);

    // Extract 3-bit indices (0-7) for 8-entry table
    const uint32_t idx_even = q_even & 0x07070707;
    const uint32_t idx_odd  = q_odd & 0x07070707;

    // Extract sign bits (bit 3 of each nibble)
    const uint32_t sign_even = (q_even >> 3) & 0x01010101;
    const uint32_t sign_odd  = (q_odd >> 3) & 0x01010101;

    // Only 2 v_perm lookups! (vs 4 in baseline)
    uint32_t val_even = __builtin_amdgcn_perm(half_values[1], half_values[0], idx_even);
    uint32_t val_odd  = __builtin_amdgcn_perm(half_values[1], half_values[0], idx_odd);

    // Generate full byte sign masks (0x00 or 0xFF)
    uint32_t sm_even = sign_even; sm_even |= sm_even<<1; sm_even |= sm_even<<2; sm_even |= sm_even<<4;
    uint32_t sm_odd  = sign_odd;  sm_odd  |= sm_odd<<1;  sm_odd  |= sm_odd<<2;  sm_odd  |= sm_odd<<4;

    // Conditional negation using: (x ^ mask) + (mask & 0x01010101)
    // This does per-byte negation but has carry issues when val=0
    // When val=0: (0 ^ 0xFF) + 1 = 0xFF + 1 = 0x100 -> overflow!

    // Actually use: result = (val ^ sm) - sm_byte (where sm_byte = sign mask as 0/1 per byte)
    // But subtraction also has borrow issues...

    // Simple approach: just do the XOR + add and hope it works for non-zero cases
    // For MXFP4, val=0 only when idx=0 or idx=8, and in both cases result should be 0
    uint32_t neg_even = (val_even ^ sm_even) + (sign_even);  // Two's complement negate
    uint32_t neg_odd  = (val_odd  ^ sm_odd)  + (sign_odd);

    // This is wrong for val=0 case, but let's test
    return make_int2(neg_even, neg_odd);
}

// Better approach: Use LUT for sign - precompute +1/-1 multipliers
// Actually simplest: just mask out the result when original val was 0
__device__ __forceinline__ int2 lookup_8entry_v2(const int q4, const uint32_t* half_values) {
    const uint32_t q_even = q4;
    const uint32_t q_odd  = (q4 >> 4);

    const uint32_t idx_even = q_even & 0x07070707;
    const uint32_t idx_odd  = q_odd & 0x07070707;
    const uint32_t sign_even = (q_even >> 3) & 0x01010101;
    const uint32_t sign_odd  = (q_odd >> 3) & 0x01010101;

    // 2 v_perm lookups
    uint32_t val_even = __builtin_amdgcn_perm(half_values[1], half_values[0], idx_even);
    uint32_t val_odd  = __builtin_amdgcn_perm(half_values[1], half_values[0], idx_odd);

    // Sign masks
    uint32_t sm_even = sign_even; sm_even |= sm_even<<1; sm_even |= sm_even<<2; sm_even |= sm_even<<4;
    uint32_t sm_odd  = sign_odd;  sm_odd  |= sm_odd<<1;  sm_odd  |= sm_odd<<2;  sm_odd  |= sm_odd<<4;

    // Detect zero values (need to mask out after negation)
    // For MXFP4: val=0 only when idx=0, so we can check idx directly
    uint32_t zero_mask_even = (idx_even == 0) ? 0xFF : 0x00;  // Per-byte? No this is wrong

    // Actually need per-byte zero detection. val==0 means all bits are 0 in that byte.
    // We can use: if any bit is set, byte is non-zero
    // zero_byte = (val == 0) per byte... complex

    // Simpler: since our values are 0,1,2,3,4,6,8,12, only 0 maps to 0
    // And idx=0 gives val=0. So: non_zero = (idx != 0) per byte
    // non_zero_even = idx_even (if idx!=0, at least one bit is set)
    // Actually idx can be 0-7, and idx=0 gives zero. All other idx give non-zero.

    // Generate non-zero mask: any bit set in idx means val is non-zero
    uint32_t nz_even = idx_even;
    nz_even |= nz_even >> 1;
    nz_even |= nz_even >> 2;  // Now each byte is 0x00 or has bits set
    // Expand to full byte
    nz_even |= nz_even << 1;
    nz_even |= nz_even << 2;
    nz_even |= nz_even << 4;  // 0x00 if idx was 0, 0xFF otherwise

    uint32_t nz_odd = idx_odd;
    nz_odd |= nz_odd >> 1;
    nz_odd |= nz_odd >> 2;
    nz_odd |= nz_odd << 1;
    nz_odd |= nz_odd << 2;
    nz_odd |= nz_odd << 4;

    // Do negation
    uint32_t neg_even = (val_even ^ sm_even) + sign_even;
    uint32_t neg_odd  = (val_odd  ^ sm_odd)  + sign_odd;

    // Mask out zeros (where original idx was 0)
    neg_even &= nz_even;
    neg_odd  &= nz_odd;

    return make_int2(neg_even, neg_odd);
}

// Cleanest approach: Use v_perm for sign application too!
// Create a sign table: {1, -1, 1, -1, 1, -1, 1, -1} ... hmm that's still complex

// What about: precompute negated values and select?
__device__ __forceinline__ int2 lookup_8entry_v3(const int q4, const uint32_t* half_values, const uint32_t* neg_half_values) {
    const uint32_t q_even = q4;
    const uint32_t q_odd  = (q4 >> 4);

    const uint32_t idx_even = q_even & 0x07070707;
    const uint32_t idx_odd  = q_odd & 0x07070707;
    const uint32_t sign_even = (q_even >> 3) & 0x01010101;
    const uint32_t sign_odd  = (q_odd >> 3) & 0x01010101;

    // 4 v_perm lookups (same as BFI version)
    uint32_t pos_even = __builtin_amdgcn_perm(half_values[1], half_values[0], idx_even);
    uint32_t pos_odd  = __builtin_amdgcn_perm(half_values[1], half_values[0], idx_odd);
    uint32_t neg_even = __builtin_amdgcn_perm(neg_half_values[1], neg_half_values[0], idx_even);
    uint32_t neg_odd  = __builtin_amdgcn_perm(neg_half_values[1], neg_half_values[0], idx_odd);

    // Sign masks
    uint32_t sm_even = sign_even; sm_even |= sm_even<<1; sm_even |= sm_even<<2; sm_even |= sm_even<<4;
    uint32_t sm_odd  = sign_odd;  sm_odd  |= sm_odd<<1;  sm_odd  |= sm_odd<<2;  sm_odd  |= sm_odd<<4;

    // Select
    uint32_t res_even = (neg_even & sm_even) | (pos_even & ~sm_even);
    uint32_t res_odd  = (neg_odd  & sm_odd)  | (pos_odd  & ~sm_odd);

    return make_int2(res_even, res_odd);
}

// Test kernels
__global__ void test_baseline_k(const int* q4_data, int2* output, const int8_t* table, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    output[idx] = lookup_baseline(q4_data[idx], (const uint32_t*)table);
}

__global__ void test_8entry_v2_k(const int* q4_data, int2* output, const int8_t* half_table, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    output[idx] = lookup_8entry_v2(q4_data[idx], (const uint32_t*)half_table);
}

__global__ void test_8entry_v3_k(const int* q4_data, int2* output,
                                  const int8_t* half_table, const int8_t* neg_half_table, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    output[idx] = lookup_8entry_v3(q4_data[idx], (const uint32_t*)half_table, (const uint32_t*)neg_half_table);
}

int main() {
    const int N = 1024 * 1024 * 16;
    const int iterations = 100;

    // Full MXFP4 table
    int8_t h_table[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
    // Half tables
    int8_t h_half[8] = {0, 1, 2, 3, 4, 6, 8, 12};
    int8_t h_neg_half[8] = {0, -1, -2, -3, -4, -6, -8, -12};

    int* d_q4;
    int2* d_output;
    int8_t *d_table, *d_half, *d_neg_half;

    hipMalloc(&d_q4, N * sizeof(int));
    hipMalloc(&d_output, N * sizeof(int2));
    hipMalloc(&d_table, 16);
    hipMalloc(&d_half, 8);
    hipMalloc(&d_neg_half, 8);
    hipMemcpy(d_table, h_table, 16, hipMemcpyHostToDevice);
    hipMemcpy(d_half, h_half, 8, hipMemcpyHostToDevice);
    hipMemcpy(d_neg_half, h_neg_half, 8, hipMemcpyHostToDevice);

    int* h_q4 = new int[N];
    for (int i = 0; i < N; i++) h_q4[i] = rand();
    hipMemcpy(d_q4, h_q4, N * sizeof(int), hipMemcpyHostToDevice);

    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);

    // Warmup & verify
    test_baseline_k<<<grid, block>>>(d_q4, d_output, d_table, N);
    hipDeviceSynchronize();

    int2* h_ref = new int2[N];
    int2* h_test = new int2[N];

    test_baseline_k<<<grid, block>>>(d_q4, d_output, d_table, N);
    hipMemcpy(h_ref, d_output, N * sizeof(int2), hipMemcpyDeviceToHost);

    // Verify 8entry_v2
    test_8entry_v2_k<<<grid, block>>>(d_q4, d_output, d_half, N);
    hipMemcpy(h_test, d_output, N * sizeof(int2), hipMemcpyDeviceToHost);
    int errors = 0;
    for (int i = 0; i < N && errors < 10; i++) {
        if (h_ref[i].x != h_test[i].x || h_ref[i].y != h_test[i].y) {
            if (errors < 3) printf("8entry_v2 mismatch %d: ref(%08x,%08x) got(%08x,%08x) q4=%08x\n",
                   i, h_ref[i].x, h_ref[i].y, h_test[i].x, h_test[i].y, h_q4[i]);
            errors++;
        }
    }
    printf("8entry_v2: %s (%d errors)\n", errors == 0 ? "PASS" : "FAIL", errors);

    // Verify 8entry_v3
    test_8entry_v3_k<<<grid, block>>>(d_q4, d_output, d_half, d_neg_half, N);
    hipMemcpy(h_test, d_output, N * sizeof(int2), hipMemcpyDeviceToHost);
    errors = 0;
    for (int i = 0; i < N && errors < 10; i++) {
        if (h_ref[i].x != h_test[i].x || h_ref[i].y != h_test[i].y) {
            if (errors < 3) printf("8entry_v3 mismatch %d: ref(%08x,%08x) got(%08x,%08x) q4=%08x\n",
                   i, h_ref[i].x, h_ref[i].y, h_test[i].x, h_test[i].y, h_q4[i]);
            errors++;
        }
    }
    printf("8entry_v3: %s (%d errors)\n", errors == 0 ? "PASS" : "FAIL", errors);

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
        test_8entry_v2_k<<<grid, block>>>(d_q4, d_output, d_half, N);
    }
    hipDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    double t_v2 = std::chrono::duration<double, std::milli>(end - start).count();

    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        test_8entry_v3_k<<<grid, block>>>(d_q4, d_output, d_half, d_neg_half, N);
    }
    hipDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    double t_v3 = std::chrono::duration<double, std::milli>(end - start).count();

    printf("\nBenchmark (%d iters, %dM lookups):\n", iterations, N/1000000);
    printf("  Baseline (6 perm):     %.2f ms\n", t_base);
    printf("  8entry_v2 (2 perm):    %.2f ms (%+.1f%%)\n", t_v2, (t_v2/t_base - 1)*100);
    printf("  8entry_v3 (4 perm):    %.2f ms (%+.1f%%)\n", t_v3, (t_v3/t_base - 1)*100);

    hipFree(d_q4);
    hipFree(d_output);
    hipFree(d_table);
    hipFree(d_half);
    hipFree(d_neg_half);
    delete[] h_q4;
    delete[] h_ref;
    delete[] h_test;

    return 0;
}
