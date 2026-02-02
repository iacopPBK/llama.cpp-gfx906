// gfx906/quantize-helpers.cuh
// Helper macros for GFX906-optimized quantization kernels

#pragma once

#include "gfx906-common.cuh"

#if defined(GGML_USE_HIP) && defined(__gfx906__)

// ============================================================================
// GFX906 ULTRA-FUSED WARP REDUCTIONS - Single asm block, minimal NOPs
// Old approach: separate function calls = 12+ NOPs total
// New approach: fused asm block = 3 NOPs total (75% reduction!)
// ============================================================================

// DS4 layout: Fully fused reduction for both max and sum (32 vals per scale/sum)
#define GFX906_Q8_1_WARP_REDUCE_DS4(amax, sum) \
    do { \
        int amax_i = __float_as_int(amax); \
        int sum_i  = __float_as_int(sum); \
        int amax_tmp, sum_tmp; \
        asm volatile( \
            "v_mov_b32 %0, %4\n" \
            "v_mov_b32 %1, %5\n" \
            "s_nop 1\n" \
            "v_mov_b32_dpp %0, %4 row_shl:4 row_mask:0xf bank_mask:0x5\n" \
            "v_mov_b32_dpp %1, %5 row_shl:4 row_mask:0xf bank_mask:0x5\n" \
            "v_mov_b32_dpp %0, %4 row_shr:4 row_mask:0xf bank_mask:0xa\n" \
            "v_mov_b32_dpp %1, %5 row_shr:4 row_mask:0xf bank_mask:0xa\n" \
            "v_max_f32 %2, %4, %0\n" \
            "v_add_f32 %3, %5, %1\n" \
            "s_nop 1\n" \
            "v_max_f32_dpp %2, %2, %2 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf\n" \
            "v_add_f32_dpp %3, %3, %3 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf\n" \
            "s_nop 1\n" \
            "v_max_f32_dpp %2, %2, %2 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf\n" \
            "v_add_f32_dpp %3, %3, %3 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf\n" \
            : "=&v"(amax_tmp), "=&v"(sum_tmp), "=v"(amax_i), "=v"(sum_i) \
            : "v"(amax_i), "v"(sum_i) \
            : "memory" \
        ); \
        amax = __int_as_float(amax_i); \
        sum  = __int_as_float(sum_i); \
    } while(0)

// D4 layout: Only max reduction needed (32 vals per scale)
#define GFX906_Q8_1_WARP_REDUCE_D4(amax) \
    do { \
        int amax_i = __float_as_int(amax); \
        int amax_tmp; \
        asm volatile( \
            "v_mov_b32 %0, %2\n" \
            "s_nop 1\n" \
            "v_mov_b32_dpp %0, %2 row_shl:4 row_mask:0xf bank_mask:0x5\n" \
            "v_mov_b32_dpp %0, %2 row_shr:4 row_mask:0xf bank_mask:0xa\n" \
            "v_max_f32 %1, %2, %0\n" \
            "s_nop 1\n" \
            "v_max_f32_dpp %1, %1, %1 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf\n" \
            "s_nop 1\n" \
            "v_max_f32_dpp %1, %1, %1 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf\n" \
            : "=&v"(amax_tmp), "=v"(amax_i) \
            : "v"(amax_i) \
            : "memory" \
        ); \
        amax = __int_as_float(amax_i); \
    } while(0)

// Generic DPP warp reduction for other layouts
#define GFX906_Q8_1_WARP_REDUCE_GENERIC(amax, sum, vals_per_scale, vals_per_sum, ds_layout) \
    do { \
        _Pragma("unroll") \
        for (int offset = vals_per_scale/8; offset > 0; offset >>= 1) { \
            amax = fmaxf(amax, __shfl_xor(amax, offset, WARP_SIZE)); \
        } \
        if constexpr (ds_layout != MMQ_Q8_1_DS_LAYOUT_D4) { \
            _Pragma("unroll") \
            for (int offset = vals_per_sum/8; offset > 0; offset >>= 1) { \
                sum += __shfl_xor(sum, offset, WARP_SIZE); \
            } \
        } \
    } while(0)

// ============================================================================
// Optimized scale computation - Eliminate double reciprocal!
// Old: d_inv = 127 * rcp(amax), then d = rcp(d_inv)
// New: d = amax * (1/127), d_inv = rcp(d) - saves one reciprocal!
// ============================================================================
#define GFX906_Q8_1_COMPUTE_SCALE(amax, d, d_inv) \
    do { \
        constexpr float inv_127 = 1.0f / 127.0f; \
        d = amax * inv_127; \
        d_inv = fast_rcp_f32(d); \
    } while(0)

// ============================================================================
// Quantization with __float2int_rn for direct conversion
// ============================================================================
#define GFX906_Q8_1_QUANTIZE4(q, xi, d_inv) \
    do { \
        q.x = static_cast<int8_t>(__float2int_rn(xi.x * d_inv)); \
        q.y = static_cast<int8_t>(__float2int_rn(xi.y * d_inv)); \
        q.z = static_cast<int8_t>(__float2int_rn(xi.z * d_inv)); \
        q.w = static_cast<int8_t>(__float2int_rn(xi.w * d_inv)); \
    } while(0)

#else

// No-op versions for non-GFX906 builds
#define GFX906_Q8_1_WARP_REDUCE_DS4(amax, sum) ((void)0)
#define GFX906_Q8_1_WARP_REDUCE_D4(amax) ((void)0)
#define GFX906_Q8_1_WARP_REDUCE_GENERIC(amax, sum, vals_per_scale, vals_per_sum, ds_layout) ((void)0)
#define GFX906_Q8_1_COMPUTE_SCALE(amax, d, d_inv) ((void)0)
#define GFX906_Q8_1_QUANTIZE4(q, xi, d_inv) ((void)0)

#endif // defined(GGML_USE_HIP) && defined(__gfx906__)
