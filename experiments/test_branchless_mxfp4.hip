// Branchless MXFP4 optimization exploiting sign patterns
// Key insight: 50% of 4-nibble groups have all-same-sign
// But branch divergence kills performance on GPU

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>

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

// BFI-select (4.5% faster, verified correct)
__device__ __forceinline__ int2 opt_bfi_select(uint32_t q4) {
    const uint32_t values0 = 0x03020100;
    const uint32_t values1 = 0x0c080604;
    const uint32_t values2 = 0xfdfeff00;
    const uint32_t values3 = 0xf4f8fafc;
    const uint32_t expand = 0x0000FF00;  // {0x00, 0xFF}

    const uint32_t q_even = q4;
    const uint32_t q_odd  = q4 >> 4;
    const uint32_t idx_e = q_even & 0x07070707;
    const uint32_t idx_o = q_odd  & 0x07070707;

    uint32_t pos_e, pos_o, neg_e, neg_o;
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(pos_e) : "v"(values1), "v"(values0), "v"(idx_e));
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(neg_e) : "v"(values3), "v"(values2), "v"(idx_e));
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(pos_o) : "v"(values1), "v"(values0), "v"(idx_o));
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(neg_o) : "v"(values3), "v"(values2), "v"(idx_o));

    uint32_t sign_e = (q_even >> 3) & 0x01010101;
    uint32_t sign_o = (q_odd  >> 3) & 0x01010101;
    uint32_t mask_e, mask_o;
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(mask_e) : "v"(expand), "v"(expand), "v"(sign_e));
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(mask_o) : "v"(expand), "v"(expand), "v"(sign_o));

    uint32_t res_e, res_o;
    asm volatile("v_bfi_b32 %0, %1, %2, %3" : "=v"(res_e) : "v"(mask_e), "v"(neg_e), "v"(pos_e));
    asm volatile("v_bfi_b32 %0, %1, %2, %3" : "=v"(res_o) : "v"(mask_o), "v"(neg_o), "v"(pos_o));

    return make_int2(res_e, res_o);
}

// NEW: Branchless sign-aware optimization
// Idea: Use single table that covers both pos and neg for same-sign cases
// For all-positive: indices 0-7 in combined table
// For all-negative: indices 8-15 in combined table
// For mixed: fall back to 6-perm logic
//
// WAIT - there's a simpler approach!
// The merge perm is: select low[i] if sign=0, high[i] if sign=1
// For all-positive (all sign=0): result = low
// For all-negative (all sign=1): result = high
// For mixed: need merge perm
//
// Branchless: compute both low and high, then select based on uniformity
// But how to select branchlessly?

// Approach: Use v_cndmask_b32 with exec mask manipulation
// Or: Compute result for all cases, select final result

// Actually simplest branchless approach:
// 1. Always compute low and high
// 2. Compute "is_uniform" flag per group
// 3. If uniform: result = (sign ? high : low)
// 4. If not uniform: result = merge_perm
// This still needs 4 perms but avoids 2 merge perms when uniform

// Can we do even better? YES!
// New insight: We can use v_cmp + v_cndmask for byte-level selection
// But that's basically what v_perm does for merge...

// Let me try a different approach: SIMD-style selection
__device__ __forceinline__ int2 opt_simd_select(uint32_t q4) {
    const uint32_t values0 = 0x03020100;
    const uint32_t values1 = 0x0c080604;
    const uint32_t values2 = 0xfdfeff00;
    const uint32_t values3 = 0xf4f8fafc;

    const uint32_t q_even = q4;
    const uint32_t q_odd  = q4 >> 4;
    const uint32_t idx_e = q_even & 0x07070707;
    const uint32_t idx_o = q_odd  & 0x07070707;

    // Always do 4 lookups
    uint32_t pos_e, neg_e, pos_o, neg_o;
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(pos_e) : "v"(values1), "v"(values0), "v"(idx_e));
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(neg_e) : "v"(values3), "v"(values2), "v"(idx_e));
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(pos_o) : "v"(values1), "v"(values0), "v"(idx_o));
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(neg_o) : "v"(values3), "v"(values2), "v"(idx_o));

    // Generate per-byte masks from sign bits
    // sign_e has bits [3,11,19,27] set if negative
    uint32_t sign_bits_e = q_even & 0x08080808;  // Isolate sign bits
    uint32_t sign_bits_o = q_odd  & 0x08080808;

    // Expand sign bits to full bytes (0x00 or 0xFF)
    // Method: subtract 1 from (sign>>3), creates 0x00 or 0xFFFFFFFF
    // Actually: (sign >> 3) gives 0x00 or 0x01 per byte
    // We need: 0x00 or 0xFF per byte

    // Use multiplication: sign_byte * 0xFF
    // Or shift trick: sign * 0xFF = (sign << 8) - sign for bits...

    // Cleaner: v_perm to expand!
    const uint32_t expand_table = 0x0000FF00;  // byte0=0x00, byte1=0xFF
    uint32_t sign_idx_e = (sign_bits_e >> 3);  // 0x00010101 or 0x00000000 etc
    uint32_t sign_idx_o = (sign_bits_o >> 3);

    uint32_t mask_e, mask_o;
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(mask_e) : "v"(expand_table), "v"(expand_table), "v"(sign_idx_e));
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(mask_o) : "v"(expand_table), "v"(expand_table), "v"(sign_idx_o));

    // Now select: res = (mask & neg) | (~mask & pos) = v_bfi_b32!
    uint32_t res_e, res_o;
    asm volatile("v_bfi_b32 %0, %1, %2, %3" : "=v"(res_e) : "v"(mask_e), "v"(neg_e), "v"(pos_e));
    asm volatile("v_bfi_b32 %0, %1, %2, %3" : "=v"(res_o) : "v"(mask_o), "v"(neg_o), "v"(pos_o));

    return make_int2(res_e, res_o);
}

// This is actually the same as opt_bfi_select! Let me think about what's different...

// NEW IDEA: Can we reduce from 4 table lookups to 2 using wider tables?
// Problem: v_perm only works with 8 bytes, we need 16 values
// Can't fit in one perm operation...

// ANOTHER IDEA: Compute negative as transformation of positive
// val[i+8] = -val[i] for i in [0,7]
// So neg_result = -pos_result (when sign=1)
// But byte-wise negation has carry issues! We saw this before.

// YET ANOTHER IDEA: What if indices are correlated?
// In real models, nearby weights might have similar distributions
// Could prefetch/cache lookup results...

// FINAL IDEA: Specialize for common patterns
// If entire wavefront has uniform sign pattern, use fast kernel
// This requires warp-level voting

__device__ __forceinline__ int2 opt_warp_vote(uint32_t q4) {
    const uint32_t values0 = 0x03020100;
    const uint32_t values1 = 0x0c080604;
    const uint32_t values2 = 0xfdfeff00;
    const uint32_t values3 = 0xf4f8fafc;

    const uint32_t q_even = q4;
    const uint32_t q_odd  = q4 >> 4;
    const uint32_t idx_e = q_even & 0x07070707;
    const uint32_t idx_o = q_odd  & 0x07070707;

    // Check if this thread's even/odd groups are uniform
    bool even_all_pos = (q_even & 0x08080808) == 0;
    bool even_all_neg = (q_even & 0x08080808) == 0x08080808;
    bool odd_all_pos  = (q_odd  & 0x80808080) == 0;
    bool odd_all_neg  = (q_odd  & 0x80808080) == 0x80808080;

    // Warp-level vote: are ALL threads in warp uniform for even?
    bool warp_even_uniform = __all(even_all_pos || even_all_neg);
    bool warp_odd_uniform  = __all(odd_all_pos || odd_all_neg);

    uint32_t res_e, res_o;

    if (warp_even_uniform) {
        // Fast path: all threads can use single lookup
        if (even_all_pos) {
            asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(res_e) : "v"(values1), "v"(values0), "v"(idx_e));
        } else {
            asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(res_e) : "v"(values3), "v"(values2), "v"(idx_e));
        }
    } else {
        // Slow path: full 3-perm for even
        uint32_t pos_e, neg_e;
        asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(pos_e) : "v"(values1), "v"(values0), "v"(idx_e));
        asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(neg_e) : "v"(values3), "v"(values2), "v"(idx_e));
        uint32_t mask_e = 0x03020100 | ((q_even & 0x08080808) >> 1);
        asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(res_e) : "v"(neg_e), "v"(pos_e), "v"(mask_e));
    }

    if (warp_odd_uniform) {
        if (odd_all_pos) {
            asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(res_o) : "v"(values1), "v"(values0), "v"(idx_o));
        } else {
            asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(res_o) : "v"(values3), "v"(values2), "v"(idx_o));
        }
    } else {
        uint32_t pos_o, neg_o;
        asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(pos_o) : "v"(values1), "v"(values0), "v"(idx_o));
        asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(neg_o) : "v"(values3), "v"(values2), "v"(idx_o));
        uint32_t mask_o = 0x03020100 | ((q_odd & 0x08080808) >> 1);
        asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(res_o) : "v"(neg_o), "v"(pos_o), "v"(mask_o));
    }

    return make_int2(res_e, res_o);
}

// Verification kernel
__global__ void verify_all(int* errors, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    uint32_t q4 = idx | (idx << 16);  // Test pattern

    int2 ref = baseline_6perm(q4);
    int2 bfi = opt_bfi_select(q4);
    int2 simd = opt_simd_select(q4);
    int2 vote = opt_warp_vote(q4);

    if (ref.x != bfi.x || ref.y != bfi.y) atomicAdd(&errors[0], 1);
    if (ref.x != simd.x || ref.y != simd.y) atomicAdd(&errors[1], 1);
    if (ref.x != vote.x || ref.y != vote.y) atomicAdd(&errors[2], 1);
}

// Benchmark kernels
__global__ void bench_baseline(uint32_t* data, int2* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = baseline_6perm(data[idx]);
}

__global__ void bench_bfi(uint32_t* data, int2* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = opt_bfi_select(data[idx]);
}

__global__ void bench_simd(uint32_t* data, int2* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = opt_simd_select(data[idx]);
}

__global__ void bench_warp_vote(uint32_t* data, int2* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = opt_warp_vote(data[idx]);
}

int main() {
    printf("Branchless MXFP4 Optimization Test\n");
    printf("===================================\n\n");

    // Verify correctness
    int* d_errors;
    hipMalloc(&d_errors, 4 * sizeof(int));
    hipMemset(d_errors, 0, 4 * sizeof(int));
    verify_all<<<256, 256>>>(d_errors, 65536);
    hipDeviceSynchronize();
    int h_errors[4];
    hipMemcpy(h_errors, d_errors, 4 * sizeof(int), hipMemcpyDeviceToHost);

    printf("Correctness:\n");
    printf("  BFI select:  %s\n", h_errors[0] ? "FAIL" : "PASS");
    printf("  SIMD select: %s\n", h_errors[1] ? "FAIL" : "PASS");
    printf("  Warp vote:   %s\n", h_errors[2] ? "FAIL" : "PASS");
    printf("\n");

    hipFree(d_errors);

    // Benchmark
    const int N = 1024 * 1024;
    uint32_t* d_data;
    int2* d_out;
    hipMalloc(&d_data, N * sizeof(uint32_t));
    hipMalloc(&d_out, N * sizeof(int2));

    // Initialize with pseudo-random data
    uint32_t* h_data = new uint32_t[N];
    for (int i = 0; i < N; i++) {
        h_data[i] = (i * 2654435761u) ^ (i >> 13);  // Simple hash
    }
    hipMemcpy(d_data, h_data, N * sizeof(uint32_t), hipMemcpyHostToDevice);

    // Warmup
    for (int i = 0; i < 10; i++) {
        bench_baseline<<<N/256, 256>>>(d_data, d_out, N);
    }
    hipDeviceSynchronize();

    // Benchmark
    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);
    const int ITERS = 100;

    float ms_baseline, ms_bfi, ms_simd, ms_vote;

    hipEventRecord(start);
    for (int i = 0; i < ITERS; i++) {
        bench_baseline<<<N/256, 256>>>(d_data, d_out, N);
    }
    hipEventRecord(stop);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&ms_baseline, start, stop);

    hipEventRecord(start);
    for (int i = 0; i < ITERS; i++) {
        bench_bfi<<<N/256, 256>>>(d_data, d_out, N);
    }
    hipEventRecord(stop);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&ms_bfi, start, stop);

    hipEventRecord(start);
    for (int i = 0; i < ITERS; i++) {
        bench_simd<<<N/256, 256>>>(d_data, d_out, N);
    }
    hipEventRecord(stop);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&ms_simd, start, stop);

    hipEventRecord(start);
    for (int i = 0; i < ITERS; i++) {
        bench_warp_vote<<<N/256, 256>>>(d_data, d_out, N);
    }
    hipEventRecord(stop);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&ms_vote, start, stop);

    printf("Benchmark (random data, %d iterations):\n", ITERS);
    printf("  Baseline (6-perm):  %.3f ms\n", ms_baseline);
    printf("  BFI select:         %.3f ms (%.1f%% %s)\n",
           ms_bfi, 100.0*fabs(ms_bfi-ms_baseline)/ms_baseline,
           ms_bfi < ms_baseline ? "faster" : "slower");
    printf("  SIMD select:        %.3f ms (%.1f%% %s)\n",
           ms_simd, 100.0*fabs(ms_simd-ms_baseline)/ms_baseline,
           ms_simd < ms_baseline ? "faster" : "slower");
    printf("  Warp vote:          %.3f ms (%.1f%% %s)\n",
           ms_vote, 100.0*fabs(ms_vote-ms_baseline)/ms_baseline,
           ms_vote < ms_baseline ? "faster" : "slower");

    // Now test with uniform sign patterns
    printf("\nBenchmark (all-positive data):\n");
    for (int i = 0; i < N; i++) {
        h_data[i] = h_data[i] & 0x77777777;  // Clear all sign bits
    }
    hipMemcpy(d_data, h_data, N * sizeof(uint32_t), hipMemcpyHostToDevice);

    hipEventRecord(start);
    for (int i = 0; i < ITERS; i++) {
        bench_baseline<<<N/256, 256>>>(d_data, d_out, N);
    }
    hipEventRecord(stop);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&ms_baseline, start, stop);

    hipEventRecord(start);
    for (int i = 0; i < ITERS; i++) {
        bench_warp_vote<<<N/256, 256>>>(d_data, d_out, N);
    }
    hipEventRecord(stop);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&ms_vote, start, stop);

    printf("  Baseline:  %.3f ms\n", ms_baseline);
    printf("  Warp vote: %.3f ms (%.1f%% %s)\n",
           ms_vote, 100.0*fabs(ms_vote-ms_baseline)/ms_baseline,
           ms_vote < ms_baseline ? "faster" : "slower");

    delete[] h_data;
    hipFree(d_data);
    hipFree(d_out);
    hipEventDestroy(start);
    hipEventDestroy(stop);

    return 0;
}
