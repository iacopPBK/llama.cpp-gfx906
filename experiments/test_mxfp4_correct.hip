// MXFP4 Correct Optimized Implementation for GFX906
// OOB zeroing DOES NOT WORK on GFX906 (returns 0xFF, not 0x00)
// Must use 6-perm with merge, but can optimize the merge step
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
#include <chrono>

//=============================================================================
// BASELINE: Original 6-perm with merge
//=============================================================================
__device__ __forceinline__
void baseline_6perm(uint32_t q, int8_t* result_even, int8_t* result_odd) {
    constexpr uint32_t tbl_lo_pos = 0x03020100u;
    constexpr uint32_t tbl_hi_pos = 0x0C080604u;
    constexpr uint32_t tbl_lo_neg = 0xFDFEFF00u;
    constexpr uint32_t tbl_hi_neg = 0xF4F8FAFCu;

    uint32_t q_even = q & 0x0F0F0F0Fu;
    uint32_t q_odd  = (q >> 4) & 0x0F0F0F0Fu;
    uint32_t idx_even = q_even & 0x07070707u;
    uint32_t idx_odd  = q_odd & 0x07070707u;

    uint32_t v_e_lo, v_e_hi, v_e_merged;
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(v_e_lo) : "v"(tbl_hi_pos), "v"(tbl_lo_pos), "v"(idx_even));
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(v_e_hi) : "v"(tbl_hi_neg), "v"(tbl_lo_neg), "v"(idx_even));
    uint32_t mask_e = 0x03020100u | ((q_even & 0x08080808u) >> 1);
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(v_e_merged) : "v"(v_e_hi), "v"(v_e_lo), "v"(mask_e));

    uint32_t v_o_lo, v_o_hi, v_o_merged;
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(v_o_lo) : "v"(tbl_hi_pos), "v"(tbl_lo_pos), "v"(idx_odd));
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(v_o_hi) : "v"(tbl_hi_neg), "v"(tbl_lo_neg), "v"(idx_odd));
    uint32_t mask_o = 0x03020100u | ((q_odd & 0x08080808u) >> 1);
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(v_o_merged) : "v"(v_o_hi), "v"(v_o_lo), "v"(mask_o));

    *reinterpret_cast<uint32_t*>(result_even) = v_e_merged;
    *reinterpret_cast<uint32_t*>(result_odd)  = v_o_merged;
}

//=============================================================================
// OPTIMIZED A: 4-perm with BFI-based selection (instead of 3rd perm)
// Use v_bfi_b32 to select between positive and negative values
//=============================================================================
__device__ __forceinline__
void opt_a_bfi(uint32_t q, int8_t* result_even, int8_t* result_odd) {
    constexpr uint32_t pos_lo = 0x03020100u;
    constexpr uint32_t pos_hi = 0x0C080604u;
    constexpr uint32_t neg_lo = 0xFDFEFF00u;
    constexpr uint32_t neg_hi = 0xF4F8FAFCu;
    constexpr uint32_t expand = 0x0000FF00u;  // {0x00, 0xFF, ...} for sign expansion

    uint32_t q_even = q & 0x0F0F0F0Fu;
    uint32_t q_odd  = (q >> 4) & 0x0F0F0F0Fu;
    uint32_t idx_e = q_even & 0x07070707u;
    uint32_t idx_o = q_odd  & 0x07070707u;

    // Lookup positive and negative magnitudes
    uint32_t pos_e, pos_o, neg_e, neg_o;
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(pos_e) : "v"(pos_hi), "v"(pos_lo), "v"(idx_e));
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(neg_e) : "v"(neg_hi), "v"(neg_lo), "v"(idx_e));
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(pos_o) : "v"(pos_hi), "v"(pos_lo), "v"(idx_o));
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(neg_o) : "v"(neg_hi), "v"(neg_lo), "v"(idx_o));

    // Extract and expand sign bits to full byte masks
    uint32_t sign_e = (q_even >> 3) & 0x01010101u;
    uint32_t sign_o = (q_odd  >> 3) & 0x01010101u;

    uint32_t mask_e, mask_o;
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(mask_e) : "v"(expand), "v"(expand), "v"(sign_e));
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(mask_o) : "v"(expand), "v"(expand), "v"(sign_o));

    // BFI select: result = (mask & neg) | (~mask & pos)
    uint32_t res_e, res_o;
    asm volatile("v_bfi_b32 %0, %1, %2, %3" : "=v"(res_e) : "v"(mask_e), "v"(neg_e), "v"(pos_e));
    asm volatile("v_bfi_b32 %0, %1, %2, %3" : "=v"(res_o) : "v"(mask_o), "v"(neg_o), "v"(pos_o));

    *reinterpret_cast<uint32_t*>(result_even) = res_e;
    *reinterpret_cast<uint32_t*>(result_odd)  = res_o;
}

//=============================================================================
// OPTIMIZED B: 6-perm with v_and_or_b32 for faster mask computation
//=============================================================================
__device__ __forceinline__
void opt_b_andor(uint32_t q, int8_t* result_even, int8_t* result_odd) {
    constexpr uint32_t tbl_lo_pos = 0x03020100u;
    constexpr uint32_t tbl_hi_pos = 0x0C080604u;
    constexpr uint32_t tbl_lo_neg = 0xFDFEFF00u;
    constexpr uint32_t tbl_hi_neg = 0xF4F8FAFCu;

    uint32_t q_even = q & 0x0F0F0F0Fu;
    uint32_t q_odd  = (q >> 4) & 0x0F0F0F0Fu;
    uint32_t idx_even = q_even & 0x07070707u;
    uint32_t idx_odd  = q_odd & 0x07070707u;

    // Use v_and_or_b32 for mask computation
    // mask = 0x03020100 | ((q & 0x08080808) >> 1)
    // Rewrite: mask = 0x03020100 + ((q >> 1) & 0x04040404)
    // Use: (q_shifted & 0x04040404) | 0x03020100 via v_and_or

    uint32_t v_e_lo, v_e_hi, v_e_merged;
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(v_e_lo) : "v"(tbl_hi_pos), "v"(tbl_lo_pos), "v"(idx_even));
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(v_e_hi) : "v"(tbl_hi_neg), "v"(tbl_lo_neg), "v"(idx_even));

    uint32_t q_e_shifted = q_even >> 1;
    uint32_t mask_e;
    asm volatile("v_and_or_b32 %0, %1, %2, %3" : "=v"(mask_e)
                 : "v"(q_e_shifted), "v"(0x04040404u), "v"(0x03020100u));
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(v_e_merged) : "v"(v_e_hi), "v"(v_e_lo), "v"(mask_e));

    uint32_t v_o_lo, v_o_hi, v_o_merged;
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(v_o_lo) : "v"(tbl_hi_pos), "v"(tbl_lo_pos), "v"(idx_odd));
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(v_o_hi) : "v"(tbl_hi_neg), "v"(tbl_lo_neg), "v"(idx_odd));

    uint32_t q_o_shifted = q_odd >> 1;
    uint32_t mask_o;
    asm volatile("v_and_or_b32 %0, %1, %2, %3" : "=v"(mask_o)
                 : "v"(q_o_shifted), "v"(0x04040404u), "v"(0x03020100u));
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(v_o_merged) : "v"(v_o_hi), "v"(v_o_lo), "v"(mask_o));

    *reinterpret_cast<uint32_t*>(result_even) = v_e_merged;
    *reinterpret_cast<uint32_t*>(result_odd)  = v_o_merged;
}

//=============================================================================
// OPTIMIZED C: Shared index computation between positive and negative
//=============================================================================
__device__ __forceinline__
void opt_c_shared(uint32_t q, int8_t* result_even, int8_t* result_odd) {
    constexpr uint32_t pos_lo = 0x03020100u;
    constexpr uint32_t pos_hi = 0x0C080604u;
    constexpr uint32_t neg_lo = 0xFDFEFF00u;
    constexpr uint32_t neg_hi = 0xF4F8FAFCu;

    uint32_t q_even = q & 0x0F0F0F0Fu;
    uint32_t q_odd  = (q >> 4) & 0x0F0F0F0Fu;

    // Shared: idx = lower 3 bits
    uint32_t idx_e = q_even & 0x07070707u;
    uint32_t idx_o = q_odd  & 0x07070707u;

    // Shared: sign bits
    uint32_t sign_e = q_even & 0x08080808u;
    uint32_t sign_o = q_odd  & 0x08080808u;

    // 4 lookups (shared idx)
    uint32_t pos_e, pos_o, neg_e, neg_o;
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(pos_e) : "v"(pos_hi), "v"(pos_lo), "v"(idx_e));
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(neg_e) : "v"(neg_hi), "v"(neg_lo), "v"(idx_e));
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(pos_o) : "v"(pos_hi), "v"(pos_lo), "v"(idx_o));
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(neg_o) : "v"(neg_hi), "v"(neg_lo), "v"(idx_o));

    // Merge masks
    uint32_t mask_e = 0x03020100u | (sign_e >> 1);
    uint32_t mask_o = 0x03020100u | (sign_o >> 1);

    // Merge with 2 more perms
    uint32_t res_e, res_o;
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(res_e) : "v"(neg_e), "v"(pos_e), "v"(mask_e));
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(res_o) : "v"(neg_o), "v"(pos_o), "v"(mask_o));

    *reinterpret_cast<uint32_t*>(result_even) = res_e;
    *reinterpret_cast<uint32_t*>(result_odd)  = res_o;
}

//=============================================================================
// Benchmarks
//=============================================================================
__global__ void bench_baseline(uint32_t* __restrict__ out, const uint32_t* __restrict__ in, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    uint32_t q = in[idx];
    int8_t re[4], ro[4];
    #pragma unroll
    for (int i = 0; i < 100; i++) {
        baseline_6perm(q, re, ro);
        q ^= *reinterpret_cast<uint32_t*>(re) ^ *reinterpret_cast<uint32_t*>(ro);
    }
    out[idx*2] = *reinterpret_cast<uint32_t*>(re);
    out[idx*2+1] = *reinterpret_cast<uint32_t*>(ro);
}

__global__ void bench_opt_a(uint32_t* __restrict__ out, const uint32_t* __restrict__ in, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    uint32_t q = in[idx];
    int8_t re[4], ro[4];
    #pragma unroll
    for (int i = 0; i < 100; i++) {
        opt_a_bfi(q, re, ro);
        q ^= *reinterpret_cast<uint32_t*>(re) ^ *reinterpret_cast<uint32_t*>(ro);
    }
    out[idx*2] = *reinterpret_cast<uint32_t*>(re);
    out[idx*2+1] = *reinterpret_cast<uint32_t*>(ro);
}

__global__ void bench_opt_b(uint32_t* __restrict__ out, const uint32_t* __restrict__ in, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    uint32_t q = in[idx];
    int8_t re[4], ro[4];
    #pragma unroll
    for (int i = 0; i < 100; i++) {
        opt_b_andor(q, re, ro);
        q ^= *reinterpret_cast<uint32_t*>(re) ^ *reinterpret_cast<uint32_t*>(ro);
    }
    out[idx*2] = *reinterpret_cast<uint32_t*>(re);
    out[idx*2+1] = *reinterpret_cast<uint32_t*>(ro);
}

__global__ void bench_opt_c(uint32_t* __restrict__ out, const uint32_t* __restrict__ in, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    uint32_t q = in[idx];
    int8_t re[4], ro[4];
    #pragma unroll
    for (int i = 0; i < 100; i++) {
        opt_c_shared(q, re, ro);
        q ^= *reinterpret_cast<uint32_t*>(re) ^ *reinterpret_cast<uint32_t*>(ro);
    }
    out[idx*2] = *reinterpret_cast<uint32_t*>(re);
    out[idx*2+1] = *reinterpret_cast<uint32_t*>(ro);
}

//=============================================================================
// Verification
//=============================================================================
__global__ void verify(int* errors) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 65536) return;
    uint32_t q = idx | (idx << 16);
    int8_t base_e[4], base_o[4];
    int8_t a_e[4], a_o[4];
    int8_t b_e[4], b_o[4];
    int8_t c_e[4], c_o[4];

    baseline_6perm(q, base_e, base_o);
    opt_a_bfi(q, a_e, a_o);
    opt_b_andor(q, b_e, b_o);
    opt_c_shared(q, c_e, c_o);

    uint32_t bv_e = *reinterpret_cast<uint32_t*>(base_e);
    uint32_t bv_o = *reinterpret_cast<uint32_t*>(base_o);

    if (*reinterpret_cast<uint32_t*>(a_e) != bv_e || *reinterpret_cast<uint32_t*>(a_o) != bv_o)
        atomicAdd(&errors[0], 1);
    if (*reinterpret_cast<uint32_t*>(b_e) != bv_e || *reinterpret_cast<uint32_t*>(b_o) != bv_o)
        atomicAdd(&errors[1], 1);
    if (*reinterpret_cast<uint32_t*>(c_e) != bv_e || *reinterpret_cast<uint32_t*>(c_o) != bv_o)
        atomicAdd(&errors[2], 1);
}

int main() {
    printf("╔═══════════════════════════════════════════════════════════════╗\n");
    printf("║  MXFP4 Correct Optimization for GFX906                        ║\n");
    printf("║  (OOB zeroing returns 0xFF, not 0x00 - must use 6-perm)       ║\n");
    printf("╚═══════════════════════════════════════════════════════════════╝\n\n");

    int* d_errors;
    hipMalloc(&d_errors, 4 * sizeof(int));
    hipMemset(d_errors, 0, 4 * sizeof(int));
    verify<<<256, 256>>>(d_errors);
    hipDeviceSynchronize();

    int h_errors[4];
    hipMemcpy(h_errors, d_errors, 4 * sizeof(int), hipMemcpyDeviceToHost);
    printf("Correctness Check:\n");
    printf("  Opt A (BFI select):      %s\n", h_errors[0] ? "FAIL" : "PASS");
    printf("  Opt B (v_and_or mask):   %s\n", h_errors[1] ? "FAIL" : "PASS");
    printf("  Opt C (shared idx):      %s\n", h_errors[2] ? "FAIL" : "PASS");
    hipFree(d_errors);

    if (h_errors[0] || h_errors[1] || h_errors[2]) {
        printf("\n⚠️  Some failed!\n\n");
    } else {
        printf("\n✓ All correct!\n\n");
    }

    const int N = 1024 * 1024;
    const int WARMUP = 10;
    const int RUNS = 50;

    uint32_t *d_in, *d_out;
    hipMalloc(&d_in, N * sizeof(uint32_t));
    hipMalloc(&d_out, N * 2 * sizeof(uint32_t));

    uint32_t* h_in = new uint32_t[N];
    for (int i = 0; i < N; i++) h_in[i] = rand();
    hipMemcpy(d_in, h_in, N * sizeof(uint32_t), hipMemcpyHostToDevice);
    delete[] h_in;

    int bs = 256, gs = (N + bs - 1) / bs;

    auto bench = [&](const char* name, auto kernel, double* ms_out) {
        for (int i = 0; i < WARMUP; i++) kernel<<<gs, bs>>>(d_out, d_in, N);
        hipDeviceSynchronize();

        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < RUNS; i++) kernel<<<gs, bs>>>(d_out, d_in, N);
        hipDeviceSynchronize();
        auto t1 = std::chrono::high_resolution_clock::now();

        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / RUNS;
        double gops = (N * 100.0) / (ms / 1000.0) / 1e9;
        *ms_out = ms;
        printf("  %-25s: %.3f ms  (%.2f Gops/s)\n", name, ms, gops);
    };

    printf("Benchmark (%d runs, %d warmup):\n", RUNS, WARMUP);
    printf("─────────────────────────────────────────────────────────\n");

    double baseline_ms, a_ms, b_ms, c_ms;
    bench("Baseline (6-perm)", bench_baseline, &baseline_ms);
    bench("Opt A (4p + 2 BFI)", bench_opt_a, &a_ms);
    bench("Opt B (6p + v_and_or)", bench_opt_b, &b_ms);
    bench("Opt C (6p shared)", bench_opt_c, &c_ms);

    printf("─────────────────────────────────────────────────────────\n\n");

    printf("Summary:\n");
    printf("  Opt A speedup: %.1f%%\n", (baseline_ms - a_ms) / baseline_ms * 100);
    printf("  Opt B speedup: %.1f%%\n", (baseline_ms - b_ms) / baseline_ms * 100);
    printf("  Opt C speedup: %.1f%%\n", (baseline_ms - c_ms) / baseline_ms * 100);

    hipFree(d_in);
    hipFree(d_out);
    return 0;
}
