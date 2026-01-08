#pragma once

// Y-tile prefetch for MMQ: issues global_load_dword for next iteration
// Hides memory latency by overlapping loads with compute

#include "gfx906-config.h"

#if defined(GGML_USE_HIP) && defined(__gfx906__)

template<int mmq_x, int mmq_tile_y_k, int nwarps, int warp_size>
static __device__ __forceinline__ int gfx906_prefetch_y_tile_v4(
    const int * __restrict__ y,
    const int ncols_y,
    const int kb0,
    const int kb0_stop,
    const int qk,
    const int blocks_per_iter) {

    if (threadIdx.y != 0) {
        return 0;
    }

    const int kb0_next = kb0 + blocks_per_iter;

    if (kb0_next >= kb0_stop) {
        return 0;
    }

    constexpr int block_q8_1_mmq_bytes = 144;
    constexpr int QK8_1_val = 32;
    const int stride_factor = qk * block_q8_1_mmq_bytes / (4 * QK8_1_val * sizeof(int));

    const int * by_next = y + ncols_y * (kb0_next * stride_factor);

    const int lane_id = threadIdx.x;
    if (lane_id >= 2) {
        return 0;
    }

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

static __device__ __forceinline__ void gfx906_prefetch_consume(int prefetch_data) {
    asm volatile(
        "v_mov_b32 %0, %0\n"
        : "+v"(prefetch_data)
    );
}

template<int mmq_x, int mmq_tile_y_k, int nwarps, int warp_size>
static __device__ __forceinline__ void gfx906_prefetch_y_tile_v2(
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

    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");

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
