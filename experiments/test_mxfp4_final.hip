// MXFP4 Final Optimized Implementation for GFX906
// Comparing original 6-perm baseline vs optimized 4-perm OOB approach
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
#include <chrono>

//=============================================================================
// BASELINE: Original 6-perm approach (from llama.cpp)
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

    // Even: 3 perms
    uint32_t v_e_lo, v_e_hi, v_e_merged;
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(v_e_lo) : "v"(tbl_hi_pos), "v"(tbl_lo_pos), "v"(idx_even));
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(v_e_hi) : "v"(tbl_hi_neg), "v"(tbl_lo_neg), "v"(idx_even));
    uint32_t mask_e = 0x03020100u | ((q_even & 0x08080808u) >> 1);
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(v_e_merged) : "v"(v_e_hi), "v"(v_e_lo), "v"(mask_e));

    // Odd: 3 perms
    uint32_t v_o_lo, v_o_hi, v_o_merged;
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(v_o_lo) : "v"(tbl_hi_pos), "v"(tbl_lo_pos), "v"(idx_odd));
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(v_o_hi) : "v"(tbl_hi_neg), "v"(tbl_lo_neg), "v"(idx_odd));
    uint32_t mask_o = 0x03020100u | ((q_odd & 0x08080808u) >> 1);
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(v_o_merged) : "v"(v_o_hi), "v"(v_o_lo), "v"(mask_o));

    *reinterpret_cast<uint32_t*>(result_even) = v_e_merged;
    *reinterpret_cast<uint32_t*>(result_odd)  = v_o_merged;
}

//=============================================================================
// OPTIMIZED: 4-perm OOB zeroing approach
// Key insight: Use v_perm's OOB zeroing (bit 7 set) to auto-select
//=============================================================================
__device__ __forceinline__
void optimized_4perm(uint32_t q, int8_t* result_even, int8_t* result_odd) {
    constexpr uint32_t pos_hi = 0x0C080604u;
    constexpr uint32_t pos_lo = 0x03020100u;
    constexpr uint32_t neg_hi = 0xF4F8FAFCu;
    constexpr uint32_t neg_lo = 0xFDFEFF00u;

    // Extract nibbles
    uint32_t q_even = q & 0x0F0F0F0Fu;
    uint32_t q_odd  = (q >> 4) & 0x0F0F0F0Fu;

    // Build indices efficiently
    // idx7 = lower 3 bits of each nibble
    uint32_t idx7_e = q_even & 0x07070707u;
    uint32_t idx7_o = q_odd  & 0x07070707u;

    // oob = sign bit shifted to bit 7 position
    uint32_t oob_e = (q_even << 4) & 0x80808080u;
    uint32_t oob_o = (q_odd  << 4) & 0x80808080u;

    // Positive index: magnitude | OOB (invalidates if sign=1)
    uint32_t pos_idx_e = idx7_e | oob_e;
    uint32_t pos_idx_o = idx7_o | oob_o;

    // Negative index: magnitude | ~OOB (invalidates if sign=0)
    uint32_t neg_idx_e = idx7_e | (oob_e ^ 0x80808080u);
    uint32_t neg_idx_o = idx7_o | (oob_o ^ 0x80808080u);

    // 4 perm lookups (OOB indices return 0)
    uint32_t pos_e, neg_e, pos_o, neg_o;
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(pos_e) : "v"(pos_hi), "v"(pos_lo), "v"(pos_idx_e));
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(neg_e) : "v"(neg_hi), "v"(neg_lo), "v"(neg_idx_e));
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(pos_o) : "v"(pos_hi), "v"(pos_lo), "v"(pos_idx_o));
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(neg_o) : "v"(neg_hi), "v"(neg_lo), "v"(neg_idx_o));

    // Merge: one is always 0, so OR gives correct result
    *reinterpret_cast<uint32_t*>(result_even) = pos_e | neg_e;
    *reinterpret_cast<uint32_t*>(result_odd)  = pos_o | neg_o;
}

//=============================================================================
// ULTRA-OPTIMIZED: Using v_and_or_b32 fused instruction
//=============================================================================
__device__ __forceinline__
void ultra_optimized(uint32_t q, int8_t* result_even, int8_t* result_odd) {
    constexpr uint32_t pos_hi = 0x0C080604u;
    constexpr uint32_t pos_lo = 0x03020100u;
    constexpr uint32_t neg_hi = 0xF4F8FAFCu;
    constexpr uint32_t neg_lo = 0xFDFEFF00u;

    uint32_t q_even = q & 0x0F0F0F0Fu;
    uint32_t q_odd  = (q >> 4) & 0x0F0F0F0Fu;

    // Use v_and_or for combined index computation
    // pos_idx = (q_shifted & 0x80808080) | (q & 0x07070707)
    uint32_t q_shifted_e = q_even << 4;
    uint32_t q_shifted_o = q_odd << 4;
    uint32_t idx7_e = q_even & 0x07070707u;
    uint32_t idx7_o = q_odd  & 0x07070707u;

    uint32_t pos_idx_e, pos_idx_o;
    asm volatile("v_and_or_b32 %0, %1, %2, %3" : "=v"(pos_idx_e)
                 : "v"(q_shifted_e), "v"(0x80808080u), "v"(idx7_e));
    asm volatile("v_and_or_b32 %0, %1, %2, %3" : "=v"(pos_idx_o)
                 : "v"(q_shifted_o), "v"(0x80808080u), "v"(idx7_o));

    uint32_t neg_idx_e = pos_idx_e ^ 0x80808080u;
    uint32_t neg_idx_o = pos_idx_o ^ 0x80808080u;

    uint32_t pos_e, neg_e, pos_o, neg_o;
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(pos_e) : "v"(pos_hi), "v"(pos_lo), "v"(pos_idx_e));
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(neg_e) : "v"(neg_hi), "v"(neg_lo), "v"(neg_idx_e));
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(pos_o) : "v"(pos_hi), "v"(pos_lo), "v"(pos_idx_o));
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(neg_o) : "v"(neg_hi), "v"(neg_lo), "v"(neg_idx_o));

    *reinterpret_cast<uint32_t*>(result_even) = pos_e | neg_e;
    *reinterpret_cast<uint32_t*>(result_odd)  = pos_o | neg_o;
}

//=============================================================================
// Benchmark kernels
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

__global__ void bench_optimized(uint32_t* __restrict__ out, const uint32_t* __restrict__ in, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    uint32_t q = in[idx];
    int8_t re[4], ro[4];
    #pragma unroll
    for (int i = 0; i < 100; i++) {
        optimized_4perm(q, re, ro);
        q ^= *reinterpret_cast<uint32_t*>(re) ^ *reinterpret_cast<uint32_t*>(ro);
    }
    out[idx*2] = *reinterpret_cast<uint32_t*>(re);
    out[idx*2+1] = *reinterpret_cast<uint32_t*>(ro);
}

__global__ void bench_ultra(uint32_t* __restrict__ out, const uint32_t* __restrict__ in, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    uint32_t q = in[idx];
    int8_t re[4], ro[4];
    #pragma unroll
    for (int i = 0; i < 100; i++) {
        ultra_optimized(q, re, ro);
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
    int8_t opt_e[4], opt_o[4];
    int8_t ultra_e[4], ultra_o[4];

    baseline_6perm(q, base_e, base_o);
    optimized_4perm(q, opt_e, opt_o);
    ultra_optimized(q, ultra_e, ultra_o);

    uint32_t bv_e = *reinterpret_cast<uint32_t*>(base_e);
    uint32_t bv_o = *reinterpret_cast<uint32_t*>(base_o);

    if (*reinterpret_cast<uint32_t*>(opt_e) != bv_e || *reinterpret_cast<uint32_t*>(opt_o) != bv_o)
        atomicAdd(&errors[0], 1);
    if (*reinterpret_cast<uint32_t*>(ultra_e) != bv_e || *reinterpret_cast<uint32_t*>(ultra_o) != bv_o)
        atomicAdd(&errors[1], 1);
}

int main() {
    printf("╔═══════════════════════════════════════════════════════════════╗\n");
    printf("║  MXFP4 Lookup Optimization - Final Results for GFX906         ║\n");
    printf("╚═══════════════════════════════════════════════════════════════╝\n\n");

    // Verify correctness
    int* d_errors;
    hipMalloc(&d_errors, 4 * sizeof(int));
    hipMemset(d_errors, 0, 4 * sizeof(int));
    verify<<<256, 256>>>(d_errors);
    hipDeviceSynchronize();

    int h_errors[4];
    hipMemcpy(h_errors, d_errors, 4 * sizeof(int), hipMemcpyDeviceToHost);
    printf("Correctness Check:\n");
    printf("  Optimized (4-perm OOB):  %s\n", h_errors[0] ? "FAIL" : "PASS");
    printf("  Ultra (v_and_or fused):  %s\n", h_errors[1] ? "FAIL" : "PASS");
    hipFree(d_errors);

    if (h_errors[0] || h_errors[1]) {
        printf("\n⚠️  Correctness failed!\n");
        return 1;
    }
    printf("\n✓ All implementations produce correct results!\n\n");

    // Benchmark
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
        printf("  %-30s: %.3f ms  (%.2f Gops/s)\n", name, ms, gops);
    };

    printf("Performance Benchmark (%d iterations, %d warmup):\n", RUNS, WARMUP);
    printf("─────────────────────────────────────────────────────────────────\n");

    double baseline_ms, opt_ms, ultra_ms;
    bench("Baseline (6-perm)", bench_baseline, &baseline_ms);
    bench("Optimized (4-perm OOB)", bench_optimized, &opt_ms);
    bench("Ultra (v_and_or fused)", bench_ultra, &ultra_ms);

    printf("─────────────────────────────────────────────────────────────────\n\n");

    double opt_speedup = (baseline_ms - opt_ms) / baseline_ms * 100;
    double ultra_speedup = (baseline_ms - ultra_ms) / baseline_ms * 100;

    printf("Summary:\n");
    printf("  Baseline:  6 v_perm + 2 merge perms = 6 perms total\n");
    printf("  Optimized: 4 v_perm + OOB zeroing   = 4 perms total (33%% fewer perms)\n\n");

    printf("  Optimized speedup:  %.1f%% faster than baseline\n", opt_speedup);
    printf("  Ultra speedup:      %.1f%% faster than baseline\n\n", ultra_speedup);

    printf("Key Optimization Techniques:\n");
    printf("  1. OOB Zeroing: v_perm returns 0 when selector byte has bit 7 set\n");
    printf("  2. XOR for neg index: neg_idx = pos_idx ^ 0x80808080\n");
    printf("  3. v_and_or_b32: fused AND+OR for index computation\n");
    printf("  4. Simple OR merge: since one of pos/neg is always 0\n\n");

    hipFree(d_in);
    hipFree(d_out);
    return 0;
}
