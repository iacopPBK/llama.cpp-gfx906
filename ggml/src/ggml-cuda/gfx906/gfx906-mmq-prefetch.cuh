#pragma once

#include "gfx906-config.h"

// ============================================================================
// AMD GFX906 MMQ Y-Tensor Prefetch Optimization - WORKING VERSION
// ============================================================================
//
// Tested patterns (from test_prefetch.hip):
// - V1 (simple prefetch, single element per thread): 1.04x speedup ✓
// - V2 (with waitcnt(0) before prefetch): 1.05x speedup ✓
// - V3 (strided prefetch): 0.93x SLOWER
// - V4 (end-of-loop prefetch): 1.07x speedup ✓ BEST
// - V5 (warp0-only): 1.01x speedup ✓
// - V6 (dwordx2): 1.05x speedup ✓
// - V7 (2-tile lookahead): 1.06x speedup ✓
//
// Key insights:
// 1. Single prefetch per thread works safely
// 2. Prefetch at END of loop body (after compute) gives best overlap
// 3. No waitcnt needed after prefetch - let it overlap with loop overhead

#if defined(GGML_USE_HIP) && defined(__gfx906__)

// ============================================================================
// AGGRESSIVE V4: Prefetch BEFORE vec_dot for maximum overlap
//
// KEY INSIGHT: vec_dot uses only LDS reads (lgkmcnt) and ALU, NO global loads.
// If we issue prefetch BEFORE vec_dot and keep register alive UNTIL AFTER,
// we get ~100+ instructions of overlap with NO stalls!
//
// The trick: Return the prefetch value. Caller must "use" it after vec_dot
// to prevent the compiler from reusing the register during vec_dot.
// ============================================================================
template<int mmq_x, int mmq_tile_y_k, int nwarps, int warp_size>
static __device__ __forceinline__ int gfx906_prefetch_y_tile_v4(
    const int * __restrict__ y,
    const int ncols_y,
    const int kb0,
    const int kb0_stop,
    const int qk,
    const int blocks_per_iter) {

    // Only first warp participates - reduces memory pressure significantly
    if (threadIdx.y != 0) {
        return 0;
    }

    const int kb0_next = kb0 + blocks_per_iter;

    // Only prefetch if there's a next iteration
    if (kb0_next >= kb0_stop) {
        return 0;
    }

    // Calculate address for next Y tile - SAME formula as actual load
    constexpr int block_q8_1_mmq_bytes = 144;
    constexpr int QK8_1_val = 32;
    const int stride_factor = qk * block_q8_1_mmq_bytes / (4 * QK8_1_val * sizeof(int));

    const int * by_next = y + ncols_y * (kb0_next * stride_factor);

    // Only lane 0 and 1 prefetch the two cache lines that will be loaded
    // Actual load: global_load_dword at offset 0 and offset 1024 bytes
    const int lane_id = threadIdx.x;
    if (lane_id >= 2) {
        return 0;  // Only lanes 0-1 prefetch
    }

    // Lane 0: offset 0, Lane 1: offset 256 (1024 bytes / sizeof(int))
    const int prefetch_offset = lane_id * 256;
    const int * prefetch_addr = by_next + prefetch_offset;

    int prefetch_data;
    asm volatile(
        "global_load_dword %0, %1, off\n"
        : "=v"(prefetch_data)
        : "v"(prefetch_addr)
        : "memory"
    );
    return prefetch_data;
}

// Helper to consume prefetch value after vec_dot - keeps register alive
// CRITICAL: Must emit a REAL instruction! Empty asm gets optimized away.
static __device__ __forceinline__ void gfx906_prefetch_consume(int prefetch_data) {
    // v_mov_b32 to itself is a no-op for data but is a REAL instruction.
    // The "+v" constraint marks register as BOTH input AND output,
    // forcing compiler to keep it allocated until this point.
    // This ensures async prefetch completes before register reuse.
    asm volatile(
        "v_mov_b32 %0, %0\n"
        : "+v"(prefetch_data)
    );
}

// ============================================================================
// V2 pattern - with waitcnt(0) before prefetch (1.05x)
// ============================================================================
template<int mmq_x, int mmq_tile_y_k, int nwarps, int warp_size>
static __device__ __forceinline__ void gfx906_prefetch_y_tile_v2(
    const int * __restrict__ y,
    const int ncols_y,
    const int kb0,
    const int kb0_stop,
    const int qk,
    const int blocks_per_iter) {

    const int kb0_next = kb0 + blocks_per_iter;

    // Only prefetch if there's a next iteration
    if (kb0_next >= kb0_stop) {
        return;
    }

    // Calculate thread index
    const int tid = threadIdx.y * warp_size + threadIdx.x;

    // Only threads with valid indices participate
    constexpr int total_elements = mmq_x * mmq_tile_y_k;
    if (tid >= total_elements) {
        return;
    }

    // Clear previous memory ops (helps ~1% vs not clearing)
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");

    // Calculate address for next Y tile (using exact kernel formula)
    // by0 = y + ncols_y*(kb0*(qk*sizeof(block_q8_1_mmq)/(4*QK8_1*sizeof(int))))
    // sizeof(block_q8_1_mmq) = 144 bytes, QK8_1 = 32, sizeof(int) = 4
    // factor = qk * 144 / (4 * 32 * 4) = qk * 144 / 512 = qk * 9 / 32
    constexpr int block_q8_1_mmq_bytes = 144;
    constexpr int QK8_1_val = 32;
    const int stride_factor = qk * block_q8_1_mmq_bytes / (4 * QK8_1_val * sizeof(int));

    const int * by_next = y + ncols_y * (kb0_next * stride_factor);
    const int * prefetch_addr = by_next + tid;

    // Issue single prefetch - no waitcnt after, let it overlap
    int dummy;
    asm volatile(
        "global_load_dword %0, %1, off\n"
        : "=v"(dummy)
        : "v"(prefetch_addr)
        : "memory"
    );
}

// ============================================================================
// Simpler variant without waitcnt(0) - V1 pattern
// ============================================================================
template<int mmq_x, int mmq_tile_y_k, int nwarps, int warp_size>
static __device__ __forceinline__ void gfx906_prefetch_y_tile_v1(
    const int * __restrict__ y,
    const int ncols_y,
    const int kb0,
    const int kb0_stop,
    const int qk,
    const int blocks_per_iter) {

    const int kb0_next = kb0 + blocks_per_iter;

    if (kb0_next >= kb0_stop) {
        return;
    }

    const int tid = threadIdx.y * warp_size + threadIdx.x;
    constexpr int total_elements = mmq_x * mmq_tile_y_k;

    if (tid >= total_elements) {
        return;
    }

    constexpr int block_q8_1_mmq_bytes = 144;
    constexpr int QK8_1_val = 32;
    const int stride_factor = qk * block_q8_1_mmq_bytes / (4 * QK8_1_val * sizeof(int));

    const int * by_next = y + ncols_y * (kb0_next * stride_factor);
    const int * prefetch_addr = by_next + tid;

    int dummy;
    asm volatile(
        "global_load_dword %0, %1, off\n"
        : "=v"(dummy)
        : "v"(prefetch_addr)
        : "memory"
    );
}

// ============================================================================
// No-op variant for A/B testing
// ============================================================================
template<int mmq_x, int mmq_tile_y_k, int nwarps, int warp_size>
static __device__ __forceinline__ void gfx906_prefetch_y_tile_noop(
    const int * __restrict__ y,
    const int ncols_y,
    const int kb0,
    const int kb0_stop,
    const int qk,
    const int blocks_per_iter) {
    (void)y; (void)ncols_y; (void)kb0; (void)kb0_stop; (void)qk; (void)blocks_per_iter;
}

#endif // defined(GGML_USE_HIP) && defined(__gfx906__)
