// gfx906/mmvq-dispatch.cuh
// Helper dispatch functions for MMVQ warp-cooperative kernels on GFX906

#pragma once

#include "../common.cuh"

// GFX906-specific warp-cooperative MMVQ kernels
#if defined(GGML_HIP_GFX906)
#include "matmul/mmvq-q4_0.cuh"
#include "matmul/mmvq-q4_1.cuh"
#include "matmul/mmvq-q8_0.cuh"
#endif

// ============================================================================
// GFX906 MMVQ Dispatch Helper
// ============================================================================
// Checks if warp-cooperative kernel should be used and dispatches it.
// Returns true if dispatch was done (caller should break), false otherwise.

#if defined(GGML_HIP_GFX906)

// Generic dispatch check - common logic for all quantization types
template<typename LaunchFunc>
static inline bool gfx906_mmvq_try_warp_coop_dispatch(
        LaunchFunc launch_fn,
        const void * vx, const void * vy, const int32_t * ids, float * dst,
        const uint32_t ncols_x, const uint32_t ncols_dst, const uint32_t nchannels_y,
        const uint32_t stride_row_x, const uint32_t stride_col_dst,
        const uint32_t channel_ratio, const uint32_t stride_channel_x,
        const uint32_t stride_channel_y, const uint32_t stride_channel_dst,
        const uint32_t sample_ratio, const uint32_t stride_sample_x,
        const uint32_t stride_sample_y, const uint32_t stride_sample_dst,
        const uint32_t nrows_x, const uint32_t nchannels_dst,
        const uint32_t nsamples_dst,
        const ggml_cuda_mm_fusion_args_device & fusion,  // fusion struct with gate, x_bias, gate_bias
        cudaStream_t stream) {

    // Check if warp-cooperative kernel is applicable:
    // - Token generation (ncols_dst == 1)
    // - No fusion operations
    // - Small matrices (MoE experts, <= 1024 cols)
    const bool has_fusion = fusion.gate != nullptr || 
                            fusion.x_bias != nullptr || 
                            fusion.gate_bias != nullptr;

    if (ncols_dst != 1 || ncols_x > 1024 || has_fusion) {
        return false;
    }

    // Compute fastdiv values
    const uint3 nchannels_y_fd   = ids ? init_fastdiv_values(nchannels_y) : make_uint3(0, 0, 0);
    const uint3 channel_ratio_fd = ids ? make_uint3(0, 0, 0) : init_fastdiv_values(channel_ratio);
    const uint3 sample_ratio_fd  = init_fastdiv_values(nsamples_dst / (nsamples_dst / sample_ratio));

    // Launch the warp-cooperative kernel
    launch_fn(vx, vy, ids, dst,
              ncols_x, nchannels_y_fd, stride_row_x, stride_col_dst,
              channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
              sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
              nrows_x, nchannels_dst, nsamples_dst, stream);

    return true;
}

// Type-specific dispatch wrappers

static inline bool gfx906_mmvq_try_q4_0(
        const void * vx, const void * vy, const int32_t * ids, float * dst,
        const uint32_t ncols_x, const uint32_t ncols_dst, const uint32_t nchannels_y,
        const uint32_t stride_row_x, const uint32_t stride_col_dst,
        const uint32_t channel_ratio, const uint32_t stride_channel_x,
        const uint32_t stride_channel_y, const uint32_t stride_channel_dst,
        const uint32_t sample_ratio, const uint32_t stride_sample_x,
        const uint32_t stride_sample_y, const uint32_t stride_sample_dst,
        const uint32_t nrows_x, const uint32_t nchannels_dst,
        const uint32_t nsamples_dst,
        const ggml_cuda_mm_fusion_args_device & fusion,
        cudaStream_t stream) {
    return gfx906_mmvq_try_warp_coop_dispatch(
        gfx906_launch_mul_mat_vec_q4_0_warp_coop,
        vx, vy, ids, dst, ncols_x, ncols_dst, nchannels_y, stride_row_x, stride_col_dst,
        channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
        sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst,
        nrows_x, nchannels_dst, nsamples_dst, fusion, stream);
}

static inline bool gfx906_mmvq_try_q4_1(
        const void * vx, const void * vy, const int32_t * ids, float * dst,
        const uint32_t ncols_x, const uint32_t ncols_dst, const uint32_t nchannels_y,
        const uint32_t stride_row_x, const uint32_t stride_col_dst,
        const uint32_t channel_ratio, const uint32_t stride_channel_x,
        const uint32_t stride_channel_y, const uint32_t stride_channel_dst,
        const uint32_t sample_ratio, const uint32_t stride_sample_x,
        const uint32_t stride_sample_y, const uint32_t stride_sample_dst,
        const uint32_t nrows_x, const uint32_t nchannels_dst,
        const uint32_t nsamples_dst,
        const ggml_cuda_mm_fusion_args_device & fusion,
        cudaStream_t stream) {
    return gfx906_mmvq_try_warp_coop_dispatch(
        gfx906_launch_mul_mat_vec_q4_1_warp_coop,
        vx, vy, ids, dst, ncols_x, ncols_dst, nchannels_y, stride_row_x, stride_col_dst,
        channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
        sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst,
        nrows_x, nchannels_dst, nsamples_dst, fusion, stream);
}

static inline bool gfx906_mmvq_try_q8_0(
        const void * vx, const void * vy, const int32_t * ids, float * dst,
        const uint32_t ncols_x, const uint32_t ncols_dst, const uint32_t nchannels_y,
        const uint32_t stride_row_x, const uint32_t stride_col_dst,
        const uint32_t channel_ratio, const uint32_t stride_channel_x,
        const uint32_t stride_channel_y, const uint32_t stride_channel_dst,
        const uint32_t sample_ratio, const uint32_t stride_sample_x,
        const uint32_t stride_sample_y, const uint32_t stride_sample_dst,
        const uint32_t nrows_x, const uint32_t nchannels_dst,
        const uint32_t nsamples_dst,
        const ggml_cuda_mm_fusion_args_device & fusion,
        cudaStream_t stream) {
    return gfx906_mmvq_try_warp_coop_dispatch(
        gfx906_launch_mul_mat_vec_q8_0_warp_coop,
        vx, vy, ids, dst, ncols_x, ncols_dst, nchannels_y, stride_row_x, stride_col_dst,
        channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
        sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst,
        nrows_x, nchannels_dst, nsamples_dst, fusion, stream);
}

#else

// No-op versions for non-GFX906 builds

struct ggml_cuda_mm_fusion_args_device;  // Forward declaration

static inline bool gfx906_mmvq_try_q4_0(
        const void *, const void *, const int32_t *, float *,
        const uint32_t, const uint32_t, const uint32_t, const uint32_t,
        const uint32_t, const uint32_t, const uint32_t, const uint32_t,
        const uint32_t, const uint32_t, const uint32_t, const uint32_t,
        const uint32_t, const uint32_t, const uint32_t, const uint32_t,
        const ggml_cuda_mm_fusion_args_device &, cudaStream_t) {
    return false;
}

static inline bool gfx906_mmvq_try_q4_1(
        const void *, const void *, const int32_t *, float *,
        const uint32_t, const uint32_t, const uint32_t, const uint32_t,
        const uint32_t, const uint32_t, const uint32_t, const uint32_t,
        const uint32_t, const uint32_t, const uint32_t, const uint32_t,
        const uint32_t, const uint32_t, const uint32_t, const uint32_t,
        const ggml_cuda_mm_fusion_args_device &, cudaStream_t) {
    return false;
}

static inline bool gfx906_mmvq_try_q8_0(
        const void *, const void *, const int32_t *, float *,
        const uint32_t, const uint32_t, const uint32_t, const uint32_t,
        const uint32_t, const uint32_t, const uint32_t, const uint32_t,
        const uint32_t, const uint32_t, const uint32_t, const uint32_t,
        const uint32_t, const uint32_t, const uint32_t, const uint32_t,
        const ggml_cuda_mm_fusion_args_device &, cudaStream_t) {
    return false;
}

#endif // defined(GGML_HIP_GFX906)
