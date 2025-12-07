// Deep analysis of MXFP4 patterns for selective fast-path optimization
// Key insight: llama.cpp processes even and odd nibbles SEPARATELY
// So we only need 4 nibbles (not 8) to have same sign for fast path

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>

// The llama.cpp 6-perm implementation (reference)
__device__ __forceinline__ int2 llamacpp_6perm(uint32_t q4) {
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

// Key insight: We can check even/odd nibbles SEPARATELY!
// Each only needs 4 nibbles to have same sign, not all 8

__device__ __forceinline__ bool even_all_positive(uint32_t q4) {
    // Even nibbles: bits [3], [11], [19], [27] of q4 must all be 0
    // These are the sign bits of nibbles at positions 0,2,4,6
    return (q4 & 0x08080808) == 0;
}

__device__ __forceinline__ bool even_all_negative(uint32_t q4) {
    return (q4 & 0x08080808) == 0x08080808;
}

__device__ __forceinline__ bool odd_all_positive(uint32_t q4) {
    // Odd nibbles: bits [7], [15], [23], [31] of q4 must all be 0
    return (q4 & 0x80808080) == 0;
}

__device__ __forceinline__ bool odd_all_negative(uint32_t q4) {
    return (q4 & 0x80808080) == 0x80808080;
}

// Optimized: Check each group and skip merge perms selectively
__device__ __forceinline__ int2 optimized_selective(uint32_t q4) {
    const uint32_t values0 = 0x03020100;
    const uint32_t values1 = 0x0c080604;
    const uint32_t values2 = 0xfdfeff00;
    const uint32_t values3 = 0xf4f8fafc;

    const uint32_t q_even = q4;
    const uint32_t q_odd  = q4 >> 4;
    const uint32_t idx_even = q_even & 0x07070707;
    const uint32_t idx_odd  = q_odd  & 0x07070707;

    uint32_t res_x, res_y;

    // Process even nibbles
    if (even_all_positive(q4)) {
        // All even nibbles are 0-7, use positive table only (1 perm)
        asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(res_x) : "v"(values1), "v"(values0), "v"(idx_even));
    } else if (even_all_negative(q4)) {
        // All even nibbles are 8-15, use negative table only (1 perm)
        asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(res_x) : "v"(values3), "v"(values2), "v"(idx_even));
    } else {
        // Mixed: need both lookups + merge (3 perms)
        uint32_t v_even_low, v_even_high;
        asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(v_even_low)  : "v"(values1), "v"(values0), "v"(idx_even));
        asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(v_even_high) : "v"(values3), "v"(values2), "v"(idx_even));
        const uint32_t mask_even = 0x03020100 | ((q_even & 0x08080808) >> 1);
        asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(res_x) : "v"(v_even_high), "v"(v_even_low), "v"(mask_even));
    }

    // Process odd nibbles
    if (odd_all_positive(q4)) {
        asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(res_y) : "v"(values1), "v"(values0), "v"(idx_odd));
    } else if (odd_all_negative(q4)) {
        asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(res_y) : "v"(values3), "v"(values2), "v"(idx_odd));
    } else {
        uint32_t v_odd_low, v_odd_high;
        asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(v_odd_low)  : "v"(values1), "v"(values0), "v"(idx_odd));
        asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(v_odd_high) : "v"(values3), "v"(values2), "v"(idx_odd));
        const uint32_t mask_odd = 0x03020100 | ((q_odd & 0x08080808) >> 1);
        asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(res_y) : "v"(v_odd_high), "v"(v_odd_low), "v"(mask_odd));
    }

    return make_int2(res_x, res_y);
}

__global__ void verify_selective(int* errors) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 65536) return;

    // Test all 16-bit patterns, replicated to 32 bits
    uint32_t q4 = idx | (idx << 16);

    int2 ref = llamacpp_6perm(q4);
    int2 opt = optimized_selective(q4);

    if (ref.x != opt.x || ref.y != opt.y) {
        atomicAdd(&errors[0], 1);
    }
}

__global__ void analyze_even_odd_patterns(uint32_t* stats) {
    // stats[0] = even_all_pos
    // stats[1] = even_all_neg
    // stats[2] = even_mixed
    // stats[3] = odd_all_pos
    // stats[4] = odd_all_neg
    // stats[5] = odd_mixed
    // stats[6] = both_fast (either even OR odd can use fast path)
    // stats[7] = both_same_sign (even and odd both same direction)
    // stats[8] = total

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 65536) return;

    uint32_t q4 = idx | (idx << 16);

    bool ep = even_all_positive(q4);
    bool en = even_all_negative(q4);
    bool op = odd_all_positive(q4);
    bool on = odd_all_negative(q4);

    if (ep) atomicAdd(&stats[0], 1);
    else if (en) atomicAdd(&stats[1], 1);
    else atomicAdd(&stats[2], 1);

    if (op) atomicAdd(&stats[3], 1);
    else if (on) atomicAdd(&stats[4], 1);
    else atomicAdd(&stats[5], 1);

    // Count cases where at least one group can use fast path
    if (ep || en || op || on) {
        atomicAdd(&stats[6], 1);
    }

    // Count cases where both groups have same sign (all pos or all neg)
    if ((ep && op) || (en && on)) {
        atomicAdd(&stats[7], 1);
    }

    atomicAdd(&stats[8], 1);
}

// Analyze the SPECIFIC values in the MXFP4 lookup table
// kvales = {0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12}
// Notice: values 0-7 map to non-negative, 8-15 map to non-positive
// But index 0 and 8 BOTH map to 0!
__global__ void analyze_zero_special_case(uint32_t* stats) {
    // stats[0] = contains index 0 (maps to 0)
    // stats[1] = contains index 8 (also maps to 0!)
    // stats[2] = all nibbles are 0 or 8 (all zeros!)

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 65536) return;

    uint32_t q4 = idx | (idx << 16);

    // Check each nibble for value 0 or 8
    bool has_0 = false, has_8 = false;
    bool all_zero_vals = true;

    for (int i = 0; i < 8; i++) {
        uint32_t nibble = (q4 >> (i * 4)) & 0xF;
        if (nibble == 0) has_0 = true;
        if (nibble == 8) has_8 = true;
        if (nibble != 0 && nibble != 8) all_zero_vals = false;
    }

    if (has_0) atomicAdd(&stats[0], 1);
    if (has_8) atomicAdd(&stats[1], 1);
    if (all_zero_vals) atomicAdd(&stats[2], 1);
}

// The BIG insight: for multiplication in dot product,
// if the VALUE is 0 (indices 0 or 8), the contribution is 0 regardless of scale!
// Can we exploit this for a fast path?
__global__ void analyze_multiply_by_zero(uint32_t* stats) {
    // What if we had a very sparse model where most weights are 0?
    // Count patterns with various zero densities
    // stats[0-8] = count of patterns with 0,1,2,3,4,5,6,7,8 zeros respectively

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 65536) return;

    uint32_t q4 = idx | (idx << 16);

    int zero_count = 0;
    for (int i = 0; i < 8; i++) {
        uint32_t nibble = (q4 >> (i * 4)) & 0xF;
        if (nibble == 0 || nibble == 8) zero_count++;
    }

    atomicAdd(&stats[zero_count], 1);
}

int main() {
    printf("MXFP4 Deep Pattern Analysis for GFX906\n");
    printf("======================================\n\n");

    // Verify correctness first
    int* d_errors;
    hipMalloc(&d_errors, sizeof(int));
    hipMemset(d_errors, 0, sizeof(int));
    verify_selective<<<256, 256>>>(d_errors);
    hipDeviceSynchronize();
    int h_errors;
    hipMemcpy(&h_errors, d_errors, sizeof(int), hipMemcpyDeviceToHost);
    printf("Selective optimization correctness: %s\n\n", h_errors ? "FAIL" : "PASS");
    hipFree(d_errors);

    // Analyze even/odd patterns
    uint32_t* d_stats;
    hipMalloc(&d_stats, 16 * sizeof(uint32_t));
    hipMemset(d_stats, 0, 16 * sizeof(uint32_t));
    analyze_even_odd_patterns<<<256, 256>>>(d_stats);
    hipDeviceSynchronize();
    uint32_t h_stats[16];
    hipMemcpy(h_stats, d_stats, 16 * sizeof(uint32_t), hipMemcpyDeviceToHost);

    uint32_t total = 65536;
    printf("Even/Odd Nibble Group Analysis (uniform distribution):\n");
    printf("-------------------------------------------------------\n");
    printf("Even nibbles (positions 0,2,4,6):\n");
    printf("  All positive (0-7):  %6u (%.1f%%)\n", h_stats[0], 100.0*h_stats[0]/total);
    printf("  All negative (8-15): %6u (%.1f%%)\n", h_stats[1], 100.0*h_stats[1]/total);
    printf("  Mixed:               %6u (%.1f%%)\n", h_stats[2], 100.0*h_stats[2]/total);
    printf("\n");
    printf("Odd nibbles (positions 1,3,5,7):\n");
    printf("  All positive (0-7):  %6u (%.1f%%)\n", h_stats[3], 100.0*h_stats[3]/total);
    printf("  All negative (8-15): %6u (%.1f%%)\n", h_stats[4], 100.0*h_stats[4]/total);
    printf("  Mixed:               %6u (%.1f%%)\n", h_stats[5], 100.0*h_stats[5]/total);
    printf("\n");
    printf("Combined analysis:\n");
    printf("  At least one group fast-pathable: %6u (%.1f%%)\n", h_stats[6], 100.0*h_stats[6]/total);
    printf("  Both groups same sign:            %6u (%.1f%%)\n", h_stats[7], 100.0*h_stats[7]/total);
    printf("\n");

    // Analyze zero special case
    hipMemset(d_stats, 0, 16 * sizeof(uint32_t));
    analyze_zero_special_case<<<256, 256>>>(d_stats);
    hipDeviceSynchronize();
    hipMemcpy(h_stats, d_stats, 16 * sizeof(uint32_t), hipMemcpyDeviceToHost);

    printf("Zero Value Analysis (indices 0 and 8 both map to value 0):\n");
    printf("----------------------------------------------------------\n");
    printf("  Contains index 0:     %6u (%.1f%%)\n", h_stats[0], 100.0*h_stats[0]/total);
    printf("  Contains index 8:     %6u (%.1f%%)\n", h_stats[1], 100.0*h_stats[1]/total);
    printf("  All nibbles zero-val: %6u (%.3f%%)\n", h_stats[2], 100.0*h_stats[2]/total);
    printf("\n");

    // Analyze zero density
    hipMemset(d_stats, 0, 16 * sizeof(uint32_t));
    analyze_multiply_by_zero<<<256, 256>>>(d_stats);
    hipDeviceSynchronize();
    hipMemcpy(h_stats, d_stats, 16 * sizeof(uint32_t), hipMemcpyDeviceToHost);

    printf("Zero Density Distribution:\n");
    printf("--------------------------\n");
    for (int i = 0; i <= 8; i++) {
        printf("  %d zeros out of 8: %6u (%.2f%%)\n", i, h_stats[i], 100.0*h_stats[i]/total);
    }
    printf("\n");

    hipFree(d_stats);

    // Calculate expected perms for various scenarios
    printf("Expected Performance Analysis:\n");
    printf("==============================\n");
    printf("\nWith uniform random distribution:\n");
    float even_fast = (100.0*(h_stats[0]+h_stats[1])/total);  // Reusing earlier stats would be wrong
    printf("  - Each 4-nibble group has 50%% * 50%% * 50%% * 50%% = 6.25%% chance all-same-sign\n");
    printf("  - But 2 groups means: P(at least one fast) = 1 - (1-0.0625)^2 * 2 ways = ~12.5%%\n");
    printf("  - Selective approach: 12.5%% of groups use 1 perm, 87.5%% use 3 perms\n");
    printf("  - Average per group: 0.125*1 + 0.875*3 = 2.75 perms (vs 3 baseline)\n");
    printf("  - Theoretical speedup: ~8%% (but branch overhead may negate this)\n");

    printf("\nFor REAL model weights:\n");
    printf("  - Sparse models: Many weights near zero (indices 0,8)\n");
    printf("  - Gaussian-like distribution: More values near center\n");
    printf("  - Layer-dependent: Some layers more uniform than others\n");
    printf("  - Need to profile actual model data!\n");

    printf("\n");
    printf("KEY INSIGHT FOR OPTIMIZATION:\n");
    printf("=============================\n");
    printf("The selective approach has branch divergence which hurts GPU performance.\n");
    printf("Better approach: Pre-compute which blocks can use fast path,\n");
    printf("then dispatch entire warps to fast or slow kernel variants.\n");
    printf("\n");
    printf("Even better: For the BFI-select approach (4.5%% faster, no branches),\n");
    printf("we avoid the branch overhead entirely while still being faster.\n");

    return 0;
}
