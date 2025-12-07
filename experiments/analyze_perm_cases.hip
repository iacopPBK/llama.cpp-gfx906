// Analyze v_perm behavior for MXFP4 lookup on GFX906
// Goal: Find patterns where faster approaches might work
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>

// The llama.cpp 6-perm implementation (reference)
__device__ __forceinline__ int2 llamacpp_6perm(uint32_t q4) {
    const uint32_t values0 = 0x03020100;  // {0, 1, 2, 3}
    const uint32_t values1 = 0x0c080604;  // {4, 6, 8, 12}
    const uint32_t values2 = 0xfdfeff00;  // {0, -1, -2, -3}
    const uint32_t values3 = 0xf4f8fafc;  // {-4, -6, -8, -12}

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

// Analysis: When is sign=0 for ALL nibbles?
// Answer: When all nibbles are in 0-7 (bit 3 = 0)
// In this case, v_even_low/v_odd_low is the answer, no merge needed!
__device__ __forceinline__ int2 fast_2perm_positive_only(uint32_t q4) {
    const uint32_t values0 = 0x03020100;
    const uint32_t values1 = 0x0c080604;

    const uint32_t q_even = q4;
    const uint32_t q_odd  = q4 >> 4;

    // Only need low table lookup when all indices are 0-7
    uint32_t res_x, res_y;
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(res_x) : "v"(values1), "v"(values0), "v"(q_even & 0x07070707));
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(res_y) : "v"(values1), "v"(values0), "v"(q_odd  & 0x07070707));

    return make_int2(res_x, res_y);
}

// Analysis: When is sign=1 for ALL nibbles?
// Answer: When all nibbles are in 8-15 (bit 3 = 1)
// In this case, v_even_high/v_odd_high is the answer!
__device__ __forceinline__ int2 fast_2perm_negative_only(uint32_t q4) {
    const uint32_t values2 = 0xfdfeff00;
    const uint32_t values3 = 0xf4f8fafc;

    const uint32_t q_even = q4;
    const uint32_t q_odd  = q4 >> 4;

    // Only need high table lookup when all indices are 8-15
    uint32_t res_x, res_y;
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(res_x) : "v"(values3), "v"(values2), "v"(q_even & 0x07070707));
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(res_y) : "v"(values3), "v"(values2), "v"(q_odd  & 0x07070707));

    return make_int2(res_x, res_y);
}

// Check if all nibbles in a 32-bit word are positive (0-7)
__device__ __forceinline__ bool all_positive(uint32_t q4) {
    // Each nibble must have bit 3 = 0
    // q4 & 0x88888888 == 0 means all sign bits are 0
    return (q4 & 0x88888888) == 0;
}

// Check if all nibbles in a 32-bit word are negative (8-15)
__device__ __forceinline__ bool all_negative(uint32_t q4) {
    // Each nibble must have bit 3 = 1
    // q4 & 0x88888888 == 0x88888888 means all sign bits are 1
    return (q4 & 0x88888888) == 0x88888888;
}

// Hybrid: Use fast path when possible, fallback to 6-perm otherwise
__device__ __forceinline__ int2 hybrid_lookup(uint32_t q4) {
    if (all_positive(q4)) {
        return fast_2perm_positive_only(q4);
    } else if (all_negative(q4)) {
        return fast_2perm_negative_only(q4);
    } else {
        return llamacpp_6perm(q4);
    }
}

__global__ void analyze_patterns(uint32_t* stats, uint32_t* sample_data, int n) {
    // stats[0] = all_positive count
    // stats[1] = all_negative count
    // stats[2] = mixed count
    // stats[3] = total

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    uint32_t q4 = sample_data[idx];
    uint32_t sign_bits = q4 & 0x88888888;

    if (sign_bits == 0) {
        atomicAdd(&stats[0], 1);  // All positive
    } else if (sign_bits == 0x88888888) {
        atomicAdd(&stats[1], 1);  // All negative
    } else {
        atomicAdd(&stats[2], 1);  // Mixed
    }
    atomicAdd(&stats[3], 1);
}

__global__ void verify_fast_paths(int* errors) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 65536) return;

    uint32_t q4 = idx | (idx << 16);

    int2 ref = llamacpp_6perm(q4);
    int2 hyb = hybrid_lookup(q4);

    if (ref.x != hyb.x || ref.y != hyb.y) {
        atomicAdd(&errors[0], 1);
    }

    // Also test the individual fast paths on appropriate inputs
    if (all_positive(q4)) {
        int2 fast_pos = fast_2perm_positive_only(q4);
        if (fast_pos.x != ref.x || fast_pos.y != ref.y) {
            atomicAdd(&errors[1], 1);
        }
    }

    if (all_negative(q4)) {
        int2 fast_neg = fast_2perm_negative_only(q4);
        if (fast_neg.x != ref.x || fast_neg.y != ref.y) {
            atomicAdd(&errors[2], 1);
        }
    }
}

// Analyze sign pattern distribution in nibbles
__global__ void detailed_sign_analysis(uint32_t* counts) {
    // counts[0-255]: distribution of sign patterns (8 bits = 8 nibbles)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 65536) return;

    uint32_t q4 = idx | (idx << 16);
    // Extract sign bits from all 8 nibbles into an 8-bit pattern
    uint32_t signs = q4 & 0x88888888;
    // Compact: bit0 = nibble0.sign, bit1 = nibble1.sign, etc.
    uint32_t pattern = 0;
    pattern |= ((signs >>  3) & 1) << 0;  // nibble 0
    pattern |= ((signs >>  7) & 1) << 1;  // nibble 1
    pattern |= ((signs >> 11) & 1) << 2;  // nibble 2
    pattern |= ((signs >> 15) & 1) << 3;  // nibble 3
    pattern |= ((signs >> 19) & 1) << 4;  // nibble 4
    pattern |= ((signs >> 23) & 1) << 5;  // nibble 5
    pattern |= ((signs >> 27) & 1) << 6;  // nibble 6
    pattern |= ((signs >> 31) & 1) << 7;  // nibble 7

    atomicAdd(&counts[pattern], 1);
}

int main() {
    printf("MXFP4 v_perm Pattern Analysis for GFX906\n");
    printf("=========================================\n\n");

    // Verify fast paths are correct
    int* d_errors;
    hipMalloc(&d_errors, 4 * sizeof(int));
    hipMemset(d_errors, 0, 4 * sizeof(int));

    verify_fast_paths<<<256, 256>>>(d_errors);
    hipDeviceSynchronize();

    int h_errors[4];
    hipMemcpy(h_errors, d_errors, 4 * sizeof(int), hipMemcpyDeviceToHost);

    printf("Fast Path Correctness:\n");
    printf("  Hybrid lookup:      %s\n", h_errors[0] ? "FAIL" : "PASS");
    printf("  Fast positive only: %s\n", h_errors[1] ? "FAIL" : "PASS");
    printf("  Fast negative only: %s\n", h_errors[2] ? "FAIL" : "PASS");
    printf("\n");

    hipFree(d_errors);

    // Analyze sign pattern distribution
    uint32_t* d_counts;
    hipMalloc(&d_counts, 256 * sizeof(uint32_t));
    hipMemset(d_counts, 0, 256 * sizeof(uint32_t));

    detailed_sign_analysis<<<256, 256>>>(d_counts);
    hipDeviceSynchronize();

    uint32_t h_counts[256];
    hipMemcpy(h_counts, d_counts, 256 * sizeof(uint32_t), hipMemcpyDeviceToHost);

    // Analyze results
    uint32_t all_pos = h_counts[0];                // Pattern 0b00000000
    uint32_t all_neg = h_counts[255];              // Pattern 0b11111111

    // Count patterns with only even nibbles having same sign
    uint32_t even_same = 0;  // Patterns where bits 0,2,4,6 are same
    uint32_t odd_same = 0;   // Patterns where bits 1,3,5,7 are same

    for (int p = 0; p < 256; p++) {
        int b0 = (p >> 0) & 1;
        int b2 = (p >> 2) & 1;
        int b4 = (p >> 4) & 1;
        int b6 = (p >> 6) & 1;
        if (b0 == b2 && b2 == b4 && b4 == b6) {
            even_same += h_counts[p];
        }

        int b1 = (p >> 1) & 1;
        int b3 = (p >> 3) & 1;
        int b5 = (p >> 5) & 1;
        int b7 = (p >> 7) & 1;
        if (b1 == b3 && b3 == b5 && b5 == b7) {
            odd_same += h_counts[p];
        }
    }

    uint32_t total = 65536;

    printf("Sign Pattern Distribution (across 65536 test values):\n");
    printf("  All positive (0-7):    %6u (%.1f%%)\n", all_pos, 100.0*all_pos/total);
    printf("  All negative (8-15):   %6u (%.1f%%)\n", all_neg, 100.0*all_neg/total);
    printf("  Mixed:                 %6u (%.1f%%)\n", total - all_pos - all_neg, 100.0*(total-all_pos-all_neg)/total);
    printf("\n");
    printf("  Even nibbles same sign: %6u (%.1f%%)\n", even_same, 100.0*even_same/total);
    printf("  Odd nibbles same sign:  %6u (%.1f%%)\n", odd_same, 100.0*odd_same/total);
    printf("\n");

    printf("Top 10 most common sign patterns:\n");
    int indices[256];
    for (int i = 0; i < 256; i++) indices[i] = i;
    // Simple bubble sort (small array)
    for (int i = 0; i < 10; i++) {
        for (int j = i+1; j < 256; j++) {
            if (h_counts[indices[j]] > h_counts[indices[i]]) {
                int tmp = indices[i];
                indices[i] = indices[j];
                indices[j] = tmp;
            }
        }
    }
    for (int i = 0; i < 10; i++) {
        int p = indices[i];
        printf("  Pattern %3d (0b%d%d%d%d%d%d%d%d): %6u (%.1f%%)\n",
               p, (p>>7)&1, (p>>6)&1, (p>>5)&1, (p>>4)&1,
                  (p>>3)&1, (p>>2)&1, (p>>1)&1, (p>>0)&1,
               h_counts[p], 100.0*h_counts[p]/total);
    }

    hipFree(d_counts);

    printf("\n");
    printf("Optimization Insight:\n");
    printf("=====================\n");
    printf("If %.1f%% of inputs are all-positive and %.1f%% are all-negative,\n",
           100.0*all_pos/total, 100.0*all_neg/total);
    printf("then %.1f%% of lookups can use fast 2-perm path (vs 6-perm).\n",
           100.0*(all_pos+all_neg)/total);
    printf("\n");
    printf("For real model data, the distribution depends on weight values.\n");
    printf("MXFP4 has symmetric distribution (sign bit), so ~50%% positive is expected.\n");

    return 0;
}
