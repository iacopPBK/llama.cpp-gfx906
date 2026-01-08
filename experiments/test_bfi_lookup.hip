// Test BFI-based selection for MXFP4 lookup
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

// 4 v_perm + 2 v_bfi: Use BFI for byte selection
// v_bfi_b32: dst = (src0 & src2) | (src1 & ~src2)
// We want: if bit3=1 use high, if bit3=0 use low
// So: dst = (high & mask) | (low & ~mask)
__device__ __forceinline__ int2 lookup_4perm_2bfi(const int q4, const uint32_t* values) {
    const uint32_t q_even = q4;
    const uint32_t q_odd  = (q4 >> 4);

    // 4 lookups (same as before)
    uint32_t v_even_low = __builtin_amdgcn_perm(values[1], values[0], q_even & 0x07070707);
    uint32_t v_odd_low = __builtin_amdgcn_perm(values[1], values[0], q_odd & 0x07070707);
    uint32_t v_even_high = __builtin_amdgcn_perm(values[3], values[2], q_even & 0x07070707);
    uint32_t v_odd_high = __builtin_amdgcn_perm(values[3], values[2], q_odd & 0x07070707);

    // Generate byte masks from bit3: 0x08 -> 0xFF, 0x00 -> 0x00
    // Extract bit3 for each byte, then expand to full byte
    uint32_t bit3_even = (q_even >> 3) & 0x01010101;  // bit3 in position 0 for each byte
    uint32_t mask_even = bit3_even * 0xFF;  // 0x01010101 * 0xFF = 0xFFFFFFFF if all 1s

    uint32_t bit3_odd = (q_odd >> 3) & 0x01010101;
    uint32_t mask_odd = bit3_odd * 0xFF;

    // BFI selection: dst = (high & mask) | (low & ~mask)
    // This should compile to v_bfi_b32 or equivalent
    uint32_t res_x = (v_even_high & mask_even) | (v_even_low & ~mask_even);
    uint32_t res_y = (v_odd_high & mask_odd) | (v_odd_low & ~mask_odd);

    return make_int2(res_x, res_y);
}

// Even more optimized: use 0 - bit for sign extension
__device__ __forceinline__ int2 lookup_4perm_2sub(const int q4, const uint32_t* values) {
    const uint32_t q_even = q4;
    const uint32_t q_odd  = (q4 >> 4);

    uint32_t v_even_low = __builtin_amdgcn_perm(values[1], values[0], q_even & 0x07070707);
    uint32_t v_odd_low = __builtin_amdgcn_perm(values[1], values[0], q_odd & 0x07070707);
    uint32_t v_even_high = __builtin_amdgcn_perm(values[3], values[2], q_even & 0x07070707);
    uint32_t v_odd_high = __builtin_amdgcn_perm(values[3], values[2], q_odd & 0x07070707);

    // Alternative mask generation: signed subtraction for sign extension
    // -1 = 0xFFFFFFFF, -0 = 0
    // But we need per-byte, not per-word...

    // Use SEXT_I8_I32 approach: subtract, then AND
    uint32_t bit3_even = (q_even >> 3) & 0x01010101;
    uint32_t mask_even = (0x01010101 - bit3_even) ^ 0xFFFFFFFF;  // Negate to get ~(0 or -1)
    mask_even &= 0xFF000000 * ((q_even >> 27) & 1) |
                 0x00FF0000 * ((q_even >> 19) & 1) |
                 0x0000FF00 * ((q_even >> 11) & 1) |
                 0x000000FF * ((q_even >> 3) & 1);

    // Actually simpler - use the multiply approach
    uint32_t b3_even = (q_even >> 3) & 0x01010101;
    // Multiply each byte by 0xFF: (b3 << 8) - b3 works per-byte
    uint32_t m_even = (b3_even << 8) - b3_even;
    m_even |= (b3_even << 16) - (b3_even << 8);  // Wrong...

    // Cleanest: just use the multiply
    b3_even = (q_even >> 3) & 0x01010101;
    m_even = b3_even | (b3_even << 1) | (b3_even << 2) | (b3_even << 3) |
             (b3_even << 4) | (b3_even << 5) | (b3_even << 6) | (b3_even << 7);

    uint32_t b3_odd = (q_odd >> 3) & 0x01010101;
    uint32_t m_odd = b3_odd | (b3_odd << 1) | (b3_odd << 2) | (b3_odd << 3) |
                     (b3_odd << 4) | (b3_odd << 5) | (b3_odd << 6) | (b3_odd << 7);

    uint32_t res_x = (v_even_high & m_even) | (v_even_low & ~m_even);
    uint32_t res_y = (v_odd_high & m_odd) | (v_odd_low & ~m_odd);

    return make_int2(res_x, res_y);
}

// Cleanest BFI approach using direct byte replication
__device__ __forceinline__ int2 lookup_4perm_clean(const int q4, const uint32_t* values) {
    const uint32_t q_even = q4;
    const uint32_t q_odd  = (q4 >> 4);

    uint32_t v_even_low = __builtin_amdgcn_perm(values[1], values[0], q_even & 0x07070707);
    uint32_t v_odd_low = __builtin_amdgcn_perm(values[1], values[0], q_odd & 0x07070707);
    uint32_t v_even_high = __builtin_amdgcn_perm(values[3], values[2], q_even & 0x07070707);
    uint32_t v_odd_high = __builtin_amdgcn_perm(values[3], values[2], q_odd & 0x07070707);

    // Use __builtin_amdgcn_ubfe to extract and replicate bits
    // Or just use arithmetic: (x & 0x08080808) becomes sign in byte position after shift
    // -((x >> 3) & 1) gives 0xFFFFFFFF or 0x00000000 per bit

    // For per-byte expansion: extract bit3, subtract to get -1 or 0 per byte
    // Trick: (0 - ((x >> 3) & 0x01010101)) only works if we handle per-byte

    // Actually the simplest is: multiply by 0xFF
    // But 0x01010101 * 0xFF = 0xFFFFFFFF, not what we want per-byte

    // Use subtraction from 0: 0 - 1 = 0xFFFFFFFF in unsigned
    // But we need per-byte arithmetic...

    // GFX906 has V_SUB_U16 but not V_SUB_U8
    // We need: for each byte, if bit3=1 then 0xFF else 0x00

    // Use the fact that: (x * 0x1FF) >> 1 with masking gives byte expansion
    // Or simpler: x | (x << 1) | (x << 2) | ... | (x << 7)

    uint32_t b3e = (q_even >> 3) & 0x01010101;
    uint32_t me = b3e;
    me |= me << 1; me |= me << 2; me |= me << 4;  // 3 ops: 0x01 -> 0xFF

    uint32_t b3o = (q_odd >> 3) & 0x01010101;
    uint32_t mo = b3o;
    mo |= mo << 1; mo |= mo << 2; mo |= mo << 4;  // 3 ops

    // BFI: dst = (high & mask) | (low & ~mask)
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

__global__ void test_4perm_2bfi(const int* q4_data, int2* output, const int8_t* table, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    const uint32_t* values = (const uint32_t*)table;
    output[idx] = lookup_4perm_2bfi(q4_data[idx], values);
}

__global__ void test_4perm_clean(const int* q4_data, int2* output, const int8_t* table, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    const uint32_t* values = (const uint32_t*)table;
    output[idx] = lookup_4perm_clean(q4_data[idx], values);
}

int main() {
    const int N = 1024 * 1024 * 16;
    const int iterations = 100;

    int8_t h_table[16] = {0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15};

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

    // Warmup
    test_6perm<<<grid, block>>>(d_q4, d_output, d_table, N);
    hipDeviceSynchronize();

    // Verify correctness
    int2* h_ref = new int2[N];
    int2* h_test = new int2[N];

    test_6perm<<<grid, block>>>(d_q4, d_output, d_table, N);
    hipMemcpy(h_ref, d_output, N * sizeof(int2), hipMemcpyDeviceToHost);

    test_4perm_2bfi<<<grid, block>>>(d_q4, d_output, d_table, N);
    hipMemcpy(h_test, d_output, N * sizeof(int2), hipMemcpyDeviceToHost);

    int errors = 0;
    for (int i = 0; i < N && errors < 10; i++) {
        if (h_ref[i].x != h_test[i].x || h_ref[i].y != h_test[i].y) {
            printf("Mismatch at %d: ref=(%08x,%08x) test=(%08x,%08x) q4=%08x\n",
                   i, h_ref[i].x, h_ref[i].y, h_test[i].x, h_test[i].y, h_q4[i]);
            errors++;
        }
    }
    if (errors == 0) printf("4perm_2bfi: PASS\n");

    test_4perm_clean<<<grid, block>>>(d_q4, d_output, d_table, N);
    hipMemcpy(h_test, d_output, N * sizeof(int2), hipMemcpyDeviceToHost);

    errors = 0;
    for (int i = 0; i < N && errors < 10; i++) {
        if (h_ref[i].x != h_test[i].x || h_ref[i].y != h_test[i].y) {
            printf("Mismatch at %d: ref=(%08x,%08x) test=(%08x,%08x) q4=%08x\n",
                   i, h_ref[i].x, h_ref[i].y, h_test[i].x, h_test[i].y, h_q4[i]);
            errors++;
        }
    }
    if (errors == 0) printf("4perm_clean: PASS\n");

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
        test_4perm_2bfi<<<grid, block>>>(d_q4, d_output, d_table, N);
    }
    hipDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    double t_bfi = std::chrono::duration<double, std::milli>(end - start).count();

    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        test_4perm_clean<<<grid, block>>>(d_q4, d_output, d_table, N);
    }
    hipDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    double t_clean = std::chrono::duration<double, std::milli>(end - start).count();

    printf("\nBenchmark (%d iterations, %d lookups):\n", iterations, N);
    printf("  6 v_perm (baseline): %.2f ms\n", t_6perm);
    printf("  4 v_perm + BFI:      %.2f ms (%+.1f%%)\n", t_bfi, (t_bfi/t_6perm - 1)*100);
    printf("  4 v_perm + clean:    %.2f ms (%+.1f%%)\n", t_clean, (t_clean/t_6perm - 1)*100);

    hipFree(d_q4);
    hipFree(d_output);
    hipFree(d_table);
    delete[] h_q4;
    delete[] h_ref;
    delete[] h_test;

    return 0;
}
