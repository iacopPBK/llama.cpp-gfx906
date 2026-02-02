// gfx906/ssm-scan-dispatch.cuh
// GFX906-optimized SSM scan kernel dispatch

#pragma once

#include "../common.cuh"

#if defined(GGML_USE_HIP)

// ============================================================================
// GFX906 SSM Scan Kernel
// ============================================================================
// Optimized version using shared memory for parallel accumulation
// and different threading model (splitH instead of c_factor)

template <int splitH, int d_state>
__global__ void __launch_bounds__(d_state, 1)
    gfx906_ssm_scan_f32_group(
        const float * __restrict__ src0, const float * __restrict__ src1, const float * __restrict__ src2,
        const float * __restrict__ src3, const float * __restrict__ src4, const float * __restrict__ src5,
        const int32_t * __restrict__ src6, float * __restrict__ dst,
        const int src0_nb2, const int src0_nb3, const int src1_nb2, const int src1_nb3, const int src2_nb1, const int src2_nb2, const int src3_nb1,
        const int src4_nb2, const int src4_nb3, const int src5_nb2, const int src5_nb3,
        const int64_t s_off, const int64_t n_head, const int64_t d_head, const int64_t n_group, const int64_t n_tok) {

    const int head_idx = (blockIdx.x * splitH) / d_head;
    const int head_off = ((blockIdx.x * splitH) % d_head) * sizeof(float);
    const int seq_idx = blockIdx.y;

    const int group_off = (head_idx / (n_head / n_group)) * d_state * sizeof(float);

    const float * s0_block = (const float *) ((const char *) src0 + src6[seq_idx] * src0_nb3 + head_idx * src0_nb2 + head_off * d_state);
    const float * x_block  = (const float *) ((const char *) src1 + (seq_idx * src1_nb3) + blockIdx.x * splitH * sizeof(float));
    const float * dt_block = (const float *) ((const char *) src2 + (seq_idx * src2_nb2) + head_idx * sizeof(float));
    const float * A_block  = (const float *) ((const char *) src3 + head_idx * src3_nb1);
    const float * B_block  = (const float *) ((const char *) src4 + (seq_idx * src4_nb3) + (group_off));
    const float * C_block  = (const float *) ((const char *) src5 + (seq_idx * src5_nb3) + (group_off));
    float *       y_block  = dst + (seq_idx * n_tok * n_head * d_head) + blockIdx.x * splitH;
    float *       s_block  = (float *) ((char *) dst + s_off + seq_idx * src0_nb3 + head_idx * src0_nb2 + head_off * d_state);

    const int stride_x  = src1_nb2 / sizeof(float);
    const int stride_dt = src2_nb2 / sizeof(float);
    const int stride_B  = src4_nb2 / sizeof(float);
    const int stride_C  = src5_nb2 / sizeof(float);
    const int stride_y  = n_head * d_head;

    float state[splitH];
    __shared__ float stateC[splitH * d_state];

#pragma unroll
    for (int j = 0; j < splitH; j++) {
        state[j] = s0_block[j * d_state + threadIdx.x];
    }

    for (int64_t i = 0; i < n_tok; i++) {
        float dt_soft_plus = dt_block[i * stride_dt];
        if (dt_soft_plus <= 20.0f) {
            dt_soft_plus = log1pf(expf(dt_soft_plus));
        }
        const float dA = expf(dt_soft_plus * A_block[0]);
        const float B = B_block[i * stride_B + threadIdx.x];
        const float C = C_block[i * stride_C + threadIdx.x];

#pragma unroll
        for (int j = 0; j < splitH; j++) {
            const float x_dt = x_block[i * stride_x + j] * dt_soft_plus;
            state[j] = (state[j] * dA) + (B * x_dt);
            stateC[j * d_state + threadIdx.x] = state[j] * C;
        }

        __syncthreads();

        // parallel accumulation for stateC
        {
            static_assert((d_state & -d_state) == d_state, "the state size has to be a power of 2");
            static_assert((splitH & -splitH) == splitH, "splitH has to be a power of 2");

#pragma unroll
            for (int w = d_state; w > WARP_SIZE; w >>= 1) {
#pragma unroll
                for (int j = 0; j < ((w >> 1) * splitH + d_state - 1) / d_state; j++) {
                    const int k = (threadIdx.x % (w >> 1)) + (d_state * (threadIdx.x / (w >> 1))) + j * d_state * (d_state / (w >> 1));
                    stateC[k] += stateC[k + (w >> 1)];
                }
                __syncthreads();
            }

            static_assert(splitH >= d_state / WARP_SIZE);

#pragma unroll
            for (int j = 0; j < splitH / (d_state / WARP_SIZE); j++) {
                float y = stateC[(threadIdx.x % WARP_SIZE) + d_state * (threadIdx.x / WARP_SIZE) + j * d_state * (d_state / WARP_SIZE)];
                y = warp_reduce_sum(y);

                if (threadIdx.x % WARP_SIZE == 0) {
                    const int k = threadIdx.x / WARP_SIZE + j * (d_state / WARP_SIZE);
                    y_block[i * stride_y + k] = y;
                }
            }
        }
    }

#pragma unroll
    for (int j = 0; j < splitH; j++) {
        s_block[j * d_state + threadIdx.x] = state[j];
    }
}

// ============================================================================
// Dispatch Function
// ============================================================================

static inline bool gfx906_ssm_scan_dispatch(
        const float * src0, const float * src1, const float * src2,
        const float * src3, const float * src4, const float * src5,
        const int32_t * src6, float * dst,
        const int src0_nb2, const int src0_nb3, const int src1_nb2, const int src1_nb3,
        const int src2_nb1, const int src2_nb2, const int src3_nb1,
        const int src4_nb2, const int src4_nb3, const int src5_nb2, const int src5_nb3,
        const int64_t s_off, const int64_t d_state, const int64_t head_dim,
        const int64_t n_head, const int64_t n_group, const int64_t n_tok, const int64_t n_seq,
        cudaStream_t stream) {

    if (src3_nb1 != sizeof(float)) {
        // Mamba-1 path - not optimized for GFX906
        return false;
    }

    // Mamba-2 path with GFX906 optimizations
    if (d_state == 128) {
        constexpr int threads = 128;
        constexpr int splitH = 16;

        if (head_dim % splitH != 0) return false;

        const dim3 blocks((n_head * head_dim + (splitH - 1)) / splitH, n_seq, 1);
        gfx906_ssm_scan_f32_group<splitH, 128><<<blocks, threads, 0, stream>>>(
            src0, src1, src2, src3, src4, src5, src6, dst,
            src0_nb2, src0_nb3, src1_nb2, src1_nb3, src2_nb1, src2_nb2, src3_nb1,
            src4_nb2, src4_nb3, src5_nb2, src5_nb3, s_off, n_head, head_dim, n_group, n_tok);
        return true;
    } else if (d_state == 256) {
        constexpr int threads = 256;
        constexpr int splitH = 16;

        if (head_dim % splitH != 0) return false;

        const dim3 blocks((n_head * head_dim + (splitH - 1)) / splitH, n_seq, 1);
        gfx906_ssm_scan_f32_group<splitH, 256><<<blocks, threads, 0, stream>>>(
            src0, src1, src2, src3, src4, src5, src6, dst,
            src0_nb2, src0_nb3, src1_nb2, src1_nb3, src2_nb1, src2_nb2, src3_nb1,
            src4_nb2, src4_nb3, src5_nb2, src5_nb3, s_off, n_head, head_dim, n_group, n_tok);
        return true;
    }

    return false;
}

#else

// No-op for non-HIP builds
static inline bool gfx906_ssm_scan_dispatch(
        const float *, const float *, const float *,
        const float *, const float *, const float *,
        const int32_t *, float *,
        const int, const int, const int, const int,
        const int, const int, const int,
        const int, const int, const int, const int,
        const int64_t, const int64_t, const int64_t,
        const int64_t, const int64_t, const int64_t, const int64_t,
        cudaStream_t) {
    return false;
}

#endif // defined(GGML_USE_HIP)
