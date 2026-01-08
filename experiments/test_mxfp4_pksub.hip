// Test MXFP4 lookup using v_pk_sub_u16 for per-byte negation
// The key insight: pack bytes into 16-bit slots, use 16-bit subtraction (no carry between pairs)

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>

__device__ __constant__ int8_t kvalues_mxfp4[16] = {
    0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12
};

// ============================================================================
// REFERENCE: Original 6 v_perm
// ============================================================================
__device__ __forceinline__ int2 lookup_reference(uint32_t q4) {
    const uint32_t *values = (const uint32_t *)kvalues_mxfp4;
    const uint32_t q_even = q4;
    const uint32_t q_odd  = (q4 >> 4);

    uint32_t v_even_low  = __builtin_amdgcn_perm(values[1], values[0], q_even & 0x07070707);
    uint32_t v_odd_low   = __builtin_amdgcn_perm(values[1], values[0], q_odd  & 0x07070707);
    uint32_t v_even_high = __builtin_amdgcn_perm(values[3], values[2], q_even & 0x07070707);
    uint32_t v_odd_high  = __builtin_amdgcn_perm(values[3], values[2], q_odd  & 0x07070707);

    uint32_t mask_even = 0x03020100 | ((q_even & 0x08080808) >> 1);
    uint32_t mask_odd  = 0x03020100 | ((q_odd  & 0x08080808) >> 1);
    uint32_t res_x = __builtin_amdgcn_perm(v_even_high, v_even_low, mask_even);
    uint32_t res_y = __builtin_amdgcn_perm(v_odd_high,  v_odd_low,  mask_odd);

    return make_int2(res_x, res_y);
}

// ============================================================================
// APPROACH: 2 v_perm (magnitude) + v_pk_sub for negation + BFI
// Key: Use 16-bit subtraction to avoid carry between bytes
// ============================================================================
__device__ __forceinline__ int2 lookup_pksub(uint32_t q4) {
    // Magnitude table (8 entries)
    const uint32_t mag_lo = 0x03020100;  // {0, 1, 2, 3}
    const uint32_t mag_hi = 0x0c080604;  // {4, 6, 8, 12}

    const uint32_t q_even = q4;
    const uint32_t q_odd  = q4 >> 4;
    const uint32_t mag_idx_even = q_even & 0x07070707;
    const uint32_t mag_idx_odd  = q_odd  & 0x07070707;

    // 2 v_perm for magnitude lookup
    uint32_t pos_even = __builtin_amdgcn_perm(mag_hi, mag_lo, mag_idx_even);
    uint32_t pos_odd  = __builtin_amdgcn_perm(mag_hi, mag_lo, mag_idx_odd);

    // Compute negative using packed 16-bit subtraction
    // Split into even/odd bytes, negate as 16-bit, recombine

    // Even bytes (0, 2) in 16-bit positions
    uint32_t pos_even_b02 = pos_even & 0x00FF00FF;
    uint32_t pos_even_b13 = (pos_even >> 8) & 0x00FF00FF;
    uint32_t pos_odd_b02 = pos_odd & 0x00FF00FF;
    uint32_t pos_odd_b13 = (pos_odd >> 8) & 0x00FF00FF;

    // v_pk_sub_u16: 16-bit subtraction, no carry between halves
    uint32_t neg_even_b02, neg_even_b13, neg_odd_b02, neg_odd_b13;
    asm volatile("v_pk_sub_u16 %0, 0, %1" : "=v"(neg_even_b02) : "v"(pos_even_b02));
    asm volatile("v_pk_sub_u16 %0, 0, %1" : "=v"(neg_even_b13) : "v"(pos_even_b13));
    asm volatile("v_pk_sub_u16 %0, 0, %1" : "=v"(neg_odd_b02) : "v"(pos_odd_b02));
    asm volatile("v_pk_sub_u16 %0, 0, %1" : "=v"(neg_odd_b13) : "v"(pos_odd_b13));

    // Recombine: mask to keep only low bytes, then merge
    uint32_t neg_even = (neg_even_b02 & 0x00FF00FF) | ((neg_even_b13 & 0x00FF00FF) << 8);
    uint32_t neg_odd  = (neg_odd_b02  & 0x00FF00FF) | ((neg_odd_b13  & 0x00FF00FF) << 8);

    // Sign extraction and expansion
    uint32_t sign_even = (q_even >> 3) & 0x01010101;
    uint32_t sign_odd  = (q_odd  >> 3) & 0x01010101;

    // Expand sign to byte mask using v_perm
    const uint32_t expand = 0x0000FF00;
    uint32_t mask_even = __builtin_amdgcn_perm(expand, expand, sign_even);
    uint32_t mask_odd  = __builtin_amdgcn_perm(expand, expand, sign_odd);

    // BFI selection
    uint32_t res_x, res_y;
    asm volatile("v_bfi_b32 %0, %1, %2, %3" : "=v"(res_x) : "v"(mask_even), "v"(neg_even), "v"(pos_even));
    asm volatile("v_bfi_b32 %0, %1, %2, %3" : "=v"(res_y) : "v"(mask_odd),  "v"(neg_odd),  "v"(pos_odd));

    return make_int2(res_x, res_y);
}

// ============================================================================
// APPROACH 2: Sign expansion via arithmetic (no v_perm for signs)
// mask = (sign << 8) - sign per 16-bit half
// ============================================================================
__device__ __forceinline__ int2 lookup_pksub_arith_sign(uint32_t q4) {
    const uint32_t mag_lo = 0x03020100;
    const uint32_t mag_hi = 0x0c080604;

    const uint32_t q_even = q4;
    const uint32_t q_odd  = q4 >> 4;
    const uint32_t mag_idx_even = q_even & 0x07070707;
    const uint32_t mag_idx_odd  = q_odd  & 0x07070707;

    // 2 v_perm for magnitude
    uint32_t pos_even = __builtin_amdgcn_perm(mag_hi, mag_lo, mag_idx_even);
    uint32_t pos_odd  = __builtin_amdgcn_perm(mag_hi, mag_lo, mag_idx_odd);

    // Compute negative via pk_sub
    uint32_t pos_even_b02 = pos_even & 0x00FF00FF;
    uint32_t pos_even_b13 = (pos_even >> 8) & 0x00FF00FF;
    uint32_t pos_odd_b02 = pos_odd & 0x00FF00FF;
    uint32_t pos_odd_b13 = (pos_odd >> 8) & 0x00FF00FF;

    uint32_t neg_even_b02, neg_even_b13, neg_odd_b02, neg_odd_b13;
    asm volatile("v_pk_sub_u16 %0, 0, %1" : "=v"(neg_even_b02) : "v"(pos_even_b02));
    asm volatile("v_pk_sub_u16 %0, 0, %1" : "=v"(neg_even_b13) : "v"(pos_even_b13));
    asm volatile("v_pk_sub_u16 %0, 0, %1" : "=v"(neg_odd_b02) : "v"(pos_odd_b02));
    asm volatile("v_pk_sub_u16 %0, 0, %1" : "=v"(neg_odd_b13) : "v"(pos_odd_b13));

    uint32_t neg_even = (neg_even_b02 & 0x00FF00FF) | ((neg_even_b13 & 0x00FF00FF) << 8);
    uint32_t neg_odd  = (neg_odd_b02  & 0x00FF00FF) | ((neg_odd_b13  & 0x00FF00FF) << 8);

    // Sign extraction
    uint32_t sign_even = (q_even >> 3) & 0x01010101;
    uint32_t sign_odd  = (q_odd  >> 3) & 0x01010101;

    // Arithmetic sign expansion: mask = (sign << 8) - sign per 16-bit
    // Split signs into 16-bit pairs
    uint32_t sign_even_b02 = sign_even & 0x00FF00FF;
    uint32_t sign_even_b13 = (sign_even >> 8) & 0x00FF00FF;
    uint32_t sign_odd_b02 = sign_odd & 0x00FF00FF;
    uint32_t sign_odd_b13 = (sign_odd >> 8) & 0x00FF00FF;

    // (sign << 8) - sign using pk_sub
    uint32_t mask_even_b02, mask_even_b13, mask_odd_b02, mask_odd_b13;
    asm volatile("v_pk_sub_u16 %0, %1, %2" : "=v"(mask_even_b02) : "v"(sign_even_b02 << 8), "v"(sign_even_b02));
    asm volatile("v_pk_sub_u16 %0, %1, %2" : "=v"(mask_even_b13) : "v"(sign_even_b13 << 8), "v"(sign_even_b13));
    asm volatile("v_pk_sub_u16 %0, %1, %2" : "=v"(mask_odd_b02) : "v"(sign_odd_b02 << 8), "v"(sign_odd_b02));
    asm volatile("v_pk_sub_u16 %0, %1, %2" : "=v"(mask_odd_b13) : "v"(sign_odd_b13 << 8), "v"(sign_odd_b13));

    // Recombine masks
    uint32_t mask_even = (mask_even_b02 & 0x00FF00FF) | ((mask_even_b13 & 0x00FF00FF) << 8);
    uint32_t mask_odd  = (mask_odd_b02  & 0x00FF00FF) | ((mask_odd_b13  & 0x00FF00FF) << 8);

    // BFI selection
    uint32_t res_x, res_y;
    asm volatile("v_bfi_b32 %0, %1, %2, %3" : "=v"(res_x) : "v"(mask_even), "v"(neg_even), "v"(pos_even));
    asm volatile("v_bfi_b32 %0, %1, %2, %3" : "=v"(res_y) : "v"(mask_odd),  "v"(neg_odd),  "v"(pos_odd));

    return make_int2(res_x, res_y);
}

// ============================================================================
// APPROACH 3: Minimal v_perm - just magnitude lookup, sign via pk_sub only
// Uses only 2 v_perm (for magnitude) + many pk_sub + BFI
// ============================================================================
__device__ __forceinline__ int2 lookup_2perm(uint32_t q4) {
    const uint32_t mag_lo = 0x03020100;
    const uint32_t mag_hi = 0x0c080604;

    const uint32_t q_even = q4;
    const uint32_t q_odd  = q4 >> 4;

    // 2 v_perm ONLY for magnitude lookup
    uint32_t pos_even = __builtin_amdgcn_perm(mag_hi, mag_lo, q_even & 0x07070707);
    uint32_t pos_odd  = __builtin_amdgcn_perm(mag_hi, mag_lo, q_odd  & 0x07070707);

    // ---- Negation via pk_sub ----
    uint32_t neg_even_b02, neg_even_b13, neg_odd_b02, neg_odd_b13;
    asm volatile("v_pk_sub_u16 %0, 0, %1" : "=v"(neg_even_b02) : "v"(pos_even & 0x00FF00FF));
    asm volatile("v_pk_sub_u16 %0, 0, %1" : "=v"(neg_even_b13) : "v"((pos_even >> 8) & 0x00FF00FF));
    asm volatile("v_pk_sub_u16 %0, 0, %1" : "=v"(neg_odd_b02) : "v"(pos_odd & 0x00FF00FF));
    asm volatile("v_pk_sub_u16 %0, 0, %1" : "=v"(neg_odd_b13) : "v"((pos_odd >> 8) & 0x00FF00FF));

    uint32_t neg_even = (neg_even_b02 & 0x00FF00FF) | ((neg_even_b13 & 0x00FF00FF) << 8);
    uint32_t neg_odd  = (neg_odd_b02  & 0x00FF00FF) | ((neg_odd_b13  & 0x00FF00FF) << 8);

    // ---- Sign expansion via pk_sub ----
    uint32_t sign_even = (q_even >> 3) & 0x01010101;
    uint32_t sign_odd  = (q_odd  >> 3) & 0x01010101;

    uint32_t mask_even_b02, mask_even_b13, mask_odd_b02, mask_odd_b13;
    uint32_t se_b02 = sign_even & 0x00FF00FF;
    uint32_t se_b13 = (sign_even >> 8) & 0x00FF00FF;
    uint32_t so_b02 = sign_odd & 0x00FF00FF;
    uint32_t so_b13 = (sign_odd >> 8) & 0x00FF00FF;

    asm volatile("v_pk_sub_u16 %0, %1, %2" : "=v"(mask_even_b02) : "v"(se_b02 << 8), "v"(se_b02));
    asm volatile("v_pk_sub_u16 %0, %1, %2" : "=v"(mask_even_b13) : "v"(se_b13 << 8), "v"(se_b13));
    asm volatile("v_pk_sub_u16 %0, %1, %2" : "=v"(mask_odd_b02) : "v"(so_b02 << 8), "v"(so_b02));
    asm volatile("v_pk_sub_u16 %0, %1, %2" : "=v"(mask_odd_b13) : "v"(so_b13 << 8), "v"(so_b13));

    uint32_t mask_even = (mask_even_b02 & 0x00FF00FF) | ((mask_even_b13 & 0x00FF00FF) << 8);
    uint32_t mask_odd  = (mask_odd_b02  & 0x00FF00FF) | ((mask_odd_b13  & 0x00FF00FF) << 8);

    // ---- BFI selection ----
    uint32_t res_x, res_y;
    asm volatile("v_bfi_b32 %0, %1, %2, %3" : "=v"(res_x) : "v"(mask_even), "v"(neg_even), "v"(pos_even));
    asm volatile("v_bfi_b32 %0, %1, %2, %3" : "=v"(res_y) : "v"(mask_odd),  "v"(neg_odd),  "v"(pos_odd));

    return make_int2(res_x, res_y);
}

// ============================================================================
// Test and benchmark
// ============================================================================
__global__ void test_correctness(uint32_t* inputs, int2* ref, int2* pksub,
                                  int2* arith, int2* twoperm, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    uint32_t q4 = inputs[idx];
    ref[idx] = lookup_reference(q4);
    pksub[idx] = lookup_pksub(q4);
    arith[idx] = lookup_pksub_arith_sign(q4);
    twoperm[idx] = lookup_2perm(q4);
}

__global__ void bench_reference(uint32_t* inputs, int2* outputs, int n, int iters) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    uint32_t q4 = inputs[idx];
    int2 result = make_int2(0, 0);
    #pragma unroll 1
    for (int i = 0; i < iters; i++) {
        int2 r = lookup_reference(q4 ^ i);
        result.x += r.x; result.y += r.y;
    }
    outputs[idx] = result;
}

__global__ void bench_pksub(uint32_t* inputs, int2* outputs, int n, int iters) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    uint32_t q4 = inputs[idx];
    int2 result = make_int2(0, 0);
    #pragma unroll 1
    for (int i = 0; i < iters; i++) {
        int2 r = lookup_pksub(q4 ^ i);
        result.x += r.x; result.y += r.y;
    }
    outputs[idx] = result;
}

__global__ void bench_2perm(uint32_t* inputs, int2* outputs, int n, int iters) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    uint32_t q4 = inputs[idx];
    int2 result = make_int2(0, 0);
    #pragma unroll 1
    for (int i = 0; i < iters; i++) {
        int2 r = lookup_2perm(q4 ^ i);
        result.x += r.x; result.y += r.y;
    }
    outputs[idx] = result;
}

int main() {
    printf("=== MXFP4 v_pk_sub Approach Test ===\n\n");

    const int N = 65536;
    const int ITERS = 10000;
    const int BLOCK = 256;
    const int GRID = (N + BLOCK - 1) / BLOCK;

    uint32_t* h_inputs = new uint32_t[N];
    int2* h_ref = new int2[N];
    int2* h_pksub = new int2[N];
    int2* h_arith = new int2[N];
    int2* h_2perm = new int2[N];

    for (int i = 0; i < N; i++) h_inputs[i] = i;

    uint32_t* d_inputs;
    int2 *d_ref, *d_pksub, *d_arith, *d_2perm;
    hipMalloc(&d_inputs, N * sizeof(uint32_t));
    hipMalloc(&d_ref, N * sizeof(int2));
    hipMalloc(&d_pksub, N * sizeof(int2));
    hipMalloc(&d_arith, N * sizeof(int2));
    hipMalloc(&d_2perm, N * sizeof(int2));
    hipMemcpy(d_inputs, h_inputs, N * sizeof(uint32_t), hipMemcpyHostToDevice);

    // Correctness test
    hipLaunchKernelGGL(test_correctness, dim3(GRID), dim3(BLOCK), 0, 0,
                       d_inputs, d_ref, d_pksub, d_arith, d_2perm, N);
    hipDeviceSynchronize();

    hipMemcpy(h_ref, d_ref, N * sizeof(int2), hipMemcpyDeviceToHost);
    hipMemcpy(h_pksub, d_pksub, N * sizeof(int2), hipMemcpyDeviceToHost);
    hipMemcpy(h_arith, d_arith, N * sizeof(int2), hipMemcpyDeviceToHost);
    hipMemcpy(h_2perm, d_2perm, N * sizeof(int2), hipMemcpyDeviceToHost);

    auto check = [&](const char* name, int2* results) {
        int errors = 0;
        for (int i = 0; i < N && errors < 3; i++) {
            if (h_ref[i].x != results[i].x || h_ref[i].y != results[i].y) {
                if (errors == 0) printf("[%s] MISMATCH:\n", name);
                printf("  i=%d (0x%04x): ref=(0x%08x,0x%08x) got=(0x%08x,0x%08x)\n",
                       i, h_inputs[i], h_ref[i].x, h_ref[i].y, results[i].x, results[i].y);
                errors++;
            }
        }
        if (errors == 0) printf("[%s] PASSED\n", name);
        else printf("[%s] FAILED (%d+ errors)\n", name, errors);
        return errors == 0;
    };

    printf("=== Correctness ===\n");
    bool pksub_ok = check("pksub (2 perm + pk_sub)", h_pksub);
    bool arith_ok = check("arith_sign (2 perm + pk_sub sign)", h_arith);
    bool twoperm_ok = check("2perm (minimal perm)", h_2perm);

    // Benchmark only correct implementations
    printf("\n=== Benchmark (%d x %d iters) ===\n", N, ITERS);

    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);

    // Warmup
    hipLaunchKernelGGL(bench_reference, dim3(GRID), dim3(BLOCK), 0, 0, d_inputs, d_ref, N, 100);
    hipDeviceSynchronize();

    // Reference
    hipEventRecord(start);
    hipLaunchKernelGGL(bench_reference, dim3(GRID), dim3(BLOCK), 0, 0, d_inputs, d_ref, N, ITERS);
    hipEventRecord(stop);
    hipEventSynchronize(stop);
    float ms_ref;
    hipEventElapsedTime(&ms_ref, start, stop);
    printf("Reference (6 v_perm):     %.2f ms\n", ms_ref);

    if (pksub_ok) {
        hipEventRecord(start);
        hipLaunchKernelGGL(bench_pksub, dim3(GRID), dim3(BLOCK), 0, 0, d_inputs, d_pksub, N, ITERS);
        hipEventRecord(stop);
        hipEventSynchronize(stop);
        float ms;
        hipEventElapsedTime(&ms, start, stop);
        printf("pksub (2 perm + pk):      %.2f ms (%.2fx)\n", ms, ms_ref / ms);
    }

    if (twoperm_ok) {
        hipEventRecord(start);
        hipLaunchKernelGGL(bench_2perm, dim3(GRID), dim3(BLOCK), 0, 0, d_inputs, d_2perm, N, ITERS);
        hipEventRecord(stop);
        hipEventSynchronize(stop);
        float ms;
        hipEventElapsedTime(&ms, start, stop);
        printf("2perm (minimal perm):     %.2f ms (%.2fx)\n", ms, ms_ref / ms);
    }

    // Cleanup
    hipFree(d_inputs); hipFree(d_ref); hipFree(d_pksub); hipFree(d_arith); hipFree(d_2perm);
    delete[] h_inputs; delete[] h_ref; delete[] h_pksub; delete[] h_arith; delete[] h_2perm;
    hipEventDestroy(start); hipEventDestroy(stop);

    printf("\n=== Done ===\n");
    return 0;
}
