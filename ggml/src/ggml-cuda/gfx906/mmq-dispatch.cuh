// gfx906/mmq-dispatch.cuh
// Helper macros for MMQ dispatch to minimize diff in mmq.cuh

#pragma once

#include "../mmq.cuh"

// ============================================================================
// MMQ_TILE_Y_K_LDS - LDS stride for Y-tile
// ============================================================================
// Note: This was added to support LDS padding experiments on GFX906.
// Testing showed that padding (stride 41) was ~3.5% slower than the original
// stride 40 due to division overhead in index calculations, despite reducing
// bank conflicts from 4-way to 1-way.
// 
// PADDING ANALYSIS RESULTS:
// Original stride 40: 40 mod 32 = 8 -> 4-way bank conflicts (9.3% LDS stalls)
// Tested: Padded stride 41 with dst_idx = l + l/40 mapping
// Result: -3.5% slower (1180 vs 1223 t/s) even with proper shared mem allocation
// Root cause: Division overhead in store loop outweighs bank conflict reduction
// The bank conflicts occur during vec_dot reads, but the overhead is in stores
// Conclusion: Keep original stride - bank conflicts are cheaper than index math
//
// This define allows both upstream and GFX906 to use the same code path,
// with the option to override on specific architectures if needed.

#ifndef MMQ_TILE_Y_K_LDS
#define MMQ_TILE_Y_K_LDS MMQ_TILE_Y_K
#endif

// ============================================================================
// Vectorized Quantized Load Helpers
// ============================================================================

#if defined(GGML_USE_HIP)

// Use GFX906 vectorized loads
#define GFX906_MMQ_LOAD_Q4_0(y_qs, base_addr, qi, u) \
    gfx906_load_q4_0_quants_vectorized(y_qs, base_addr, qi, u)

#define GFX906_MMQ_LOAD_Q4_1(y_qs, base_addr, qi, u) \
    gfx906_load_q4_1_quants_vectorized(y_qs, base_addr, qi, u)

#else

// Upstream: use scalar loads
#define GFX906_MMQ_LOAD_Q4_0(y_qs, base_addr, qi, u) \
    do { \
        _Pragma("unroll") \
        for (int l = 0; l < VDR_Q4_0_Q8_1_MMQ; ++l) { \
            u[2*l+0] = y_qs[base_addr +  l]; \
            u[2*l+1] = y_qs[base_addr + (l + qi)]; \
        } \
    } while(0)

#define GFX906_MMQ_LOAD_Q4_1(y_qs, base_addr, qi, u) \
    do { \
        _Pragma("unroll") \
        for (int l = 0; l < VDR_Q4_1_Q8_1_MMQ; ++l) { \
            u[2*l+0] = y_qs[base_addr +  l]; \
            u[2*l+1] = y_qs[base_addr + (l + qi)]; \
        } \
    } while(0)

#endif // defined(GGML_USE_HIP)

// ============================================================================
// Q8_0 Tile Load Macros (Software Pipelining)
// ============================================================================

#if defined(GGML_USE_HIP) && defined(__gfx906__)

// GFX906: Use software pipelining with separate load/store phases
#define MMQ_LOAD_TILES_Q8_0_OPTIMIZED(cache_size, nrows, nwarps, threads_per_row, need_check, \
    x, kbx0, stride, i_max, txi, kbx, kqsx, qs0_cache, qs1_cache, i_slot_cache) \
    GFX906_LOAD_TILES_Q8_0_ASYNC(cache_size, nrows, nwarps, threads_per_row, need_check, \
        x, kbx0, stride, i_max, txi, kbx, kqsx, qs0_cache, qs1_cache, i_slot_cache)

#define MMQ_STORE_TILES_Q8_0_MMA(cache_size, x_qs, qs0_cache, qs1_cache, i_slot_cache, txi) \
    GFX906_STORE_TILES_Q8_0_LDS_MMA(cache_size, x_qs, qs0_cache, qs1_cache, i_slot_cache, txi)

#define MMQ_STORE_TILES_Q8_0_LEGACY(cache_size, x_qs, qs0_cache, qs1_cache, i_slot_cache, txi) \
    GFX906_STORE_TILES_Q8_0_LDS_LEGACY(cache_size, x_qs, qs0_cache, qs1_cache, i_slot_cache, txi)

#else

// Upstream/NON-GFX906: Use standard loop
#define MMQ_LOAD_TILES_Q8_0_OPTIMIZED(cache_size, nrows, nwarps, threads_per_row, need_check, \
    x, kbx0, stride, i_max, txi, kbx, kqsx, qs0_cache, qs1_cache, i_slot_cache) \
    (void)qs0_cache; (void)qs1_cache; (void)i_slot_cache

#define MMQ_STORE_TILES_Q8_0_MMA(cache_size, x_qs, qs0_cache, qs1_cache, i_slot_cache, txi) \
    (void)cache_size; (void)x_qs; (void)qs0_cache; (void)qs1_cache; (void)i_slot_cache; (void)txi

#define MMQ_STORE_TILES_Q8_0_LEGACY(cache_size, x_qs, qs0_cache, qs1_cache, i_slot_cache, txi) \
    (void)cache_size; (void)x_qs; (void)qs0_cache; (void)qs1_cache; (void)i_slot_cache; (void)txi

#endif // defined(GGML_USE_HIP) && defined(__gfx906__)

// ============================================================================
// MXFP4 Software Pipelining
// ============================================================================

#if defined(GGML_USE_HIP) && defined(__gfx906__)

// GFX906: Software pipelining for MXFP4 - separate load and dequant phases
#define MMQ_LOAD_MXFP4_PIPELINED_BEGIN(loop_iters, nrows, nwarps, threads_per_row, need_check, \
    x, kbx0, stride, i_max, kbx, kqsx, aux_q4_cache, i_cache) \
    do { \
        _Pragma("unroll") \
        for (int iter = 0; iter < (loop_iters > 16 ? 16 : loop_iters); iter++) { \
            const int i0 = iter * nrows * nwarps; \
            int i = i0 + (nrows == 1 ? threadIdx.y : threadIdx.y*nrows + threadIdx.x/threads_per_row); \
            if (need_check) { \
                i = min(i, i_max); \
            } \
            const block_mxfp4 * bxi = (const block_mxfp4 *) x + kbx0 + i*stride + kbx; \
            aux_q4_cache[iter] = get_int_b1(bxi->qs, kqsx); \
            i_cache[iter] = i; \
        } \
    } while(0)

#define MMQ_LOAD_MXFP4_PIPELINED_END_MMA(loop_iters, x_qs, aux_q4_cache, i_cache, kbx, kqsx) \
    do { \
        const int k0 = kbx * (2 * QI_MXFP4) + kqsx; \
        _Pragma("unroll") \
        for (int iter = 0; iter < (loop_iters > 16 ? 16 : loop_iters); iter++) { \
            const int2 v = get_int_from_mxfp4_table(aux_q4_cache[iter]); \
            const int i = i_cache[iter]; \
            x_qs[i*MMQ_MMA_TILE_X_K_Q8_1 + k0 + 0] = v.x; \
            x_qs[i*MMQ_MMA_TILE_X_K_Q8_1 + k0 + QI_MXFP4] = v.y; \
        } \
    } while(0)

#define MMQ_LOAD_MXFP4_PIPELINED_END_DP4A(loop_iters, x_qs, aux_q4_cache, i_cache, kbx, kqsx) \
    do { \
        const int k0 = kbx * (2 * QI_MXFP4) + kqsx; \
        _Pragma("unroll") \
        for (int iter = 0; iter < (loop_iters > 16 ? 16 : loop_iters); iter++) { \
            const int2 v = get_int_from_mxfp4_table(aux_q4_cache[iter]); \
            const int i = i_cache[iter]; \
            x_qs[i*(2*MMQ_TILE_NE_K + 1) + k0 + 0] = v.x; \
            x_qs[i*(2*MMQ_TILE_NE_K + 1) + k0 + QI_MXFP4] = v.y; \
        } \
    } while(0)

#else

// No pipelining for non-GFX906
#define MMQ_LOAD_MXFP4_PIPELINED_BEGIN(loop_iters, nrows, nwarps, threads_per_row, need_check, \
    x, kbx0, stride, i_max, kbx, kqsx, aux_q4_cache, i_cache) \
    (void)aux_q4_cache; (void)i_cache

#define MMQ_LOAD_MXFP4_PIPELINED_END_MMA(loop_iters, x_qs, aux_q4_cache, i_cache, kbx, kqsx) \
    (void)loop_iters; (void)x_qs; (void)aux_q4_cache; (void)i_cache; (void)kbx; (void)kqsx

#define MMQ_LOAD_MXFP4_PIPELINED_END_DP4A(loop_iters, x_qs, aux_q4_cache, i_cache, kbx, kqsx) \
    (void)loop_iters; (void)x_qs; (void)aux_q4_cache; (void)i_cache; (void)kbx; (void)kqsx

#endif // defined(GGML_USE_HIP) && defined(__gfx906__)

// ============================================================================
// LDS Write Conflict Avoidance
// ============================================================================

// GFX906 optimization: Avoid LDS write conflicts in need_check path.
// Original code clamped i to i_max, causing all out-of-bounds threads to
// write to the SAME location (tile[i_max*...]) - serializing LDS writes.
// Fix: Each thread writes to its ORIGINAL slot; out-of-bounds write zeros.

#if defined(GGML_USE_HIP)

#define MMQ_CALC_I_SLOT_AND_READ(i_slot, i_read, oob, i0, nrows, nwarps, threads_per_row, need_check, i_max) \
    do { \
        i_slot = i0 + (nrows == 1 ? threadIdx.y : threadIdx.y*nrows + threadIdx.x/threads_per_row); \
        i_read = need_check ? min(i_slot, i_max) : i_slot; \
        oob = need_check && (i_slot > i_max); \
    } while(0)

#else

#define MMQ_CALC_I_SLOT_AND_READ(i_slot, i_read, oob, i0, nrows, nwarps, threads_per_row, need_check, i_max) \
    do { \
        i_slot = i0 + (nrows == 1 ? threadIdx.y : threadIdx.y*nrows + threadIdx.x/threads_per_row); \
        i_read = need_check ? min(i_slot, i_max) : i_slot; \
        oob = false; \
    } while(0)

#endif // defined(GGML_USE_HIP)
