/*
 * GFX906-Optimized Flash Attention Tile Kernel (F16)
 *
 * Specialized implementation for AMD MI50/MI60 (gfx906) GPUs.
 * Key optimizations:
 * - 64-thread wavefront support (vs 32-thread warps)
 * - Register blocking for 8x memory access reduction
 * - Bank conflict elimination with strategic padding
 * - Native dual-FP16 V_DOT2_F32_F16 instructions
 * - Scalar half operations for better stability
 *
 * Target: Head dimension D=128 only
 * Architecture: GFX906 (MI50/MI60)
 */

#include "common.cuh"
#include "fattn-common.cuh"
#include "fattn-tile-f16-gfx906.cuh"

#define FATTN_KQ_STRIDE_TILE_F16 128

// GFX906 shared memory bank conflict optimization
// 32-bank architecture requires strategic padding for stride-access patterns
constexpr int GFX906_KV_PADDING = 48;  // KV_tmp: D+48 = 176 (optimal alignment)
constexpr int GFX906_Q_PADDING = 32;   // Q_h: D+32 = 160 (secondary optimization)

// GFX906-only kernel optimized for 64-thread wavefronts
template<int D, int ncols, int nwarps, bool use_logit_softcap> // D == head size
__launch_bounds__(nwarps*64, 2)
static __global__ void flash_attn_tile_ext_f16_warp(
        const char * __restrict__ Q,
        const char * __restrict__ K,
        const char * __restrict__ V,
        const char * __restrict__ mask,
        const char * __restrict__ sinks,
        const int  * __restrict__ KV_max,
        float      * __restrict__ dst,
        float2     * __restrict__ dst_meta,
        const float scale,
        const float max_bias,
        const float m0,
        const float m1,
        const uint32_t n_head_log2,
        const float logit_softcap,
        const int32_t ne00, const int32_t ne01, const int32_t ne02, const int32_t ne03,
                            const int32_t nb01, const int32_t nb02, const int32_t nb03,
        const int32_t ne10, const int32_t ne11, const int32_t ne12, const int32_t ne13,
                            const int32_t nb11, const int32_t nb12, const int64_t nb13,
                            const int32_t nb21, const int32_t nb22, const int64_t nb23,
        const int32_t ne31, const int32_t ne32, const int32_t ne33,
                            const int32_t nb31, const int32_t nb32, const int64_t nb33) {
#if defined(FLASH_ATTN_AVAILABLE) && defined(FP16_AVAILABLE)

    // GFX906 only supports D=128, skip other dimensions for logit softcap
    if (use_logit_softcap && D != 128) {
        NO_DEVICE_CODE;
        return;
    }

    //In this kernel Q, K, V are matrices while i, j, k are matrix indices.

    const int ic0 = blockIdx.x * ncols; // Index of the Q/QKV column to work on.

    const int sequence = blockIdx.z / ne02;
    const int head = blockIdx.z - sequence*ne02;
    const int gqa_ratio = ne02 / ne12; // With grouped query attention there are > 1 Q matrices per K, V matrix.
    const float2 * Q_f2   = (const float2 *) (Q    + nb03* sequence         + nb02* head              + nb01*ic0);
    const half2  * K_h2   = (const half2  *) (K    + nb13* sequence         + nb12*(head / gqa_ratio));
    const half2  * V_h2   = (const half2  *) (V    + nb13* sequence         + nb12*(head / gqa_ratio)); // K and V have same shape
    const half   * maskh  = (const half   *) (mask  + nb33*(sequence % ne33)                          + nb31*ic0);
    const float  * sinksf = (const float  *) (sinks);

    const int stride_KV2 = nb11 / sizeof(half2);

    const float slopef = get_alibi_slope(max_bias, head, n_head_log2, m0, m1);
    const half  slopeh = __float2half(slopef);

    static_assert(D % 128 == 0, "D not divisible by 128 (2*64 wavefront size).");

    __shared__ half KQ[ncols*FATTN_KQ_STRIDE_TILE_F16];
    // Remove half2 casting - use scalar half operations

    // GFX906-optimized with strategic bank padding
    __shared__ half KV_tmp[FATTN_KQ_STRIDE_TILE_F16][D + GFX906_KV_PADDING];  // 128+64=192 (6*32)
    half kqmax[ncols/nwarps];
#pragma unroll
    for (int j0 = 0; j0 < ncols; j0 += nwarps) {
        kqmax[j0/nwarps] = -HALF_MAX_HALF;
    }
    half kqsum[ncols/nwarps] = {0.0f};

    half VKQ[ncols/nwarps][D/64] = {{0.0f}};

    // Q matrix with bank padding to avoid conflicts during dot product
    __shared__ half Q_h[ncols][D + GFX906_Q_PADDING];  // 128+32=160 (5*32)
#pragma unroll
    for (int j0 = 0; j0 < ncols; j0 += nwarps) {
        const int j = j0 + threadIdx.y;

#pragma unroll
        for (int i0 = 0; i0 < D; i0 += 64) {
            const int i = i0 + threadIdx.x;

            // Convert from float2 to individual half values
            const int q_idx = i / 2;
            const float2 tmp = ic0 + j < ne01 ? Q_f2[j*(nb01/sizeof(float2)) + q_idx] : make_float2(0.0f, 0.0f);
            if (i % 2 == 0) {
                Q_h[j][i] = __float2half(scale * tmp.x);
            } else {
                Q_h[j][i] = __float2half(scale * tmp.y);
            }
        }
    }

    __syncthreads();

    const int k_VKQ_max = KV_max ? KV_max[sequence*gridDim.x + blockIdx.x] : ne11;
    for (int k_VKQ_0 = blockIdx.y*FATTN_KQ_STRIDE_TILE_F16; k_VKQ_0 < k_VKQ_max; k_VKQ_0 += gridDim.y*FATTN_KQ_STRIDE_TILE_F16) {
        // Calculate KQ tile and keep track of new maximum KQ values:

        half kqmax_new[ncols/nwarps];
#pragma unroll
        for (int j = 0; j < ncols/nwarps; ++j) {
            kqmax_new[j] = kqmax[j];
        }

        // Load K matrix with half2 unpacking
#pragma unroll
        for (int i_KQ_0 = 0; i_KQ_0 < FATTN_KQ_STRIDE_TILE_F16; i_KQ_0 += nwarps) {
            const int i_KQ = i_KQ_0 + threadIdx.y;

#pragma unroll
            for (int k_KQ_0 = 0; k_KQ_0 < D; k_KQ_0 += 64) {
                const int k_KQ = k_KQ_0 + threadIdx.x;

                // Load half2 and branchless unpack
                const int k_idx = k_KQ / 2;
                const half2 k_h2 = K_h2[int64_t(k_VKQ_0 + i_KQ)*stride_KV2 + k_idx];
                
                // Branchless half2 unpacking
                const half k_low = __low2half(k_h2);
                const half k_high = __high2half(k_h2);
                KV_tmp[i_KQ][k_KQ] = (k_KQ & 1) ? k_high : k_low;
            }
        }

        __syncthreads();

        // QK dot product with register blocking optimization for GFX906
        float sum_accumulator[FATTN_KQ_STRIDE_TILE_F16/64][ncols/nwarps] = {{0.0f}};
        
        // Register blocking: Load 8 K/Q pairs per iteration to reduce shared memory accesses by 8x
        constexpr int BLOCK_SIZE = 8;
        static_assert(D % (2 * BLOCK_SIZE) == 0, "D must be divisible by 2×BLOCK_SIZE for register blocking");
        
#pragma unroll 4
        for (int k_block = 0; k_block < D; k_block += 2 * BLOCK_SIZE) {
            // Register arrays: Load once, use multiple times
            uint32_t K_block[FATTN_KQ_STRIDE_TILE_F16/64][BLOCK_SIZE];
            uint32_t Q_block[ncols/nwarps][BLOCK_SIZE];
            
            // Load 8 dual-FP16 pairs into register blocks
#pragma unroll
            for (int block_offset = 0; block_offset < BLOCK_SIZE; ++block_offset) {
                const int k_dual = k_block + block_offset * 2;
                
#pragma unroll
                for (int i_KQ_0 = 0; i_KQ_0 < FATTN_KQ_STRIDE_TILE_F16; i_KQ_0 += 64) {
                    const int i_KQ = i_KQ_0 + threadIdx.x;
                    K_block[i_KQ_0/64][block_offset] = *reinterpret_cast<const uint32_t*>(&KV_tmp[i_KQ][k_dual]);
                }
                
#pragma unroll
                for (int j_KQ_0 = 0; j_KQ_0 < ncols; j_KQ_0 += nwarps) {
                    const int j_KQ = j_KQ_0 + threadIdx.y;
                    Q_block[j_KQ_0/nwarps][block_offset] = *reinterpret_cast<const uint32_t*>(&Q_h[j_KQ][k_dual]);
                }
            }
            
            // Compute 8 MAC operations using register-cached data
#pragma unroll
            for (int i_KQ_0 = 0; i_KQ_0 < FATTN_KQ_STRIDE_TILE_F16; i_KQ_0 += 64) {
#pragma unroll
                for (int j_KQ_0 = 0; j_KQ_0 < ncols; j_KQ_0 += nwarps) {
                    // 8 consecutive MACs from register-cached data
#pragma unroll
                    for (int block_offset = 0; block_offset < BLOCK_SIZE; ++block_offset) {
                        sum_accumulator[i_KQ_0/64][j_KQ_0/nwarps] = gfx906_dot2_f16(
                            K_block[i_KQ_0/64][block_offset],
                            Q_block[j_KQ_0/nwarps][block_offset],
                            sum_accumulator[i_KQ_0/64][j_KQ_0/nwarps]
                        );
                    }
                }
            }
        }

#pragma unroll
        for (int i_KQ_0 = 0; i_KQ_0 < FATTN_KQ_STRIDE_TILE_F16; i_KQ_0 += 64) {
            const int i_KQ = i_KQ_0 + threadIdx.x;

#pragma unroll
            for (int j_KQ_0 = 0; j_KQ_0 < ncols; j_KQ_0 += nwarps) {
                const int j_KQ = j_KQ_0 + threadIdx.y;

                half sum;
                if (use_logit_softcap) {
                    // Use float accumulator directly for logit softcap
                    const float tmp = sum_accumulator[i_KQ_0/64][j_KQ_0/nwarps];
                    sum = __float2half(logit_softcap * tanhf(tmp));
                } else {
                    // Convert float accumulator to half
                    sum = __float2half(sum_accumulator[i_KQ_0/64][j_KQ_0/nwarps]);
                }
                sum += mask ? slopeh*maskh[j_KQ*ne11 + k_VKQ_0 + i_KQ] : __float2half(0.0f);

                kqmax_new[j_KQ_0/nwarps] = ggml_cuda_hmax(kqmax_new[j_KQ_0/nwarps], sum);

                KQ[j_KQ*FATTN_KQ_STRIDE_TILE_F16 + i_KQ] = sum;
            }
        }

        __syncthreads();

#pragma unroll
        for (int j0 = 0; j0 < ncols; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

            kqmax_new[j0/nwarps] = warp_reduce_max<64>(kqmax_new[j0/nwarps]);
            const half KQ_max_scale = hexp(kqmax[j0/nwarps] - kqmax_new[j0/nwarps]);
            kqmax[j0/nwarps] = kqmax_new[j0/nwarps];

            half kqsum_add = 0.0f;  // Like F32 kernel - accumulate then add
#pragma unroll
            for (int i0 = 0; i0 < FATTN_KQ_STRIDE_TILE_F16; i0 += 64) {
                const int i = i0 + threadIdx.x;

                const half diff = KQ[j*FATTN_KQ_STRIDE_TILE_F16 + i] - kqmax[j0/nwarps];
                const half val = hexp(diff);
                kqsum_add += val;  // Scalar addition like F32
                KQ[j*FATTN_KQ_STRIDE_TILE_F16 + i] = val;
            }
            
            kqsum[j0/nwarps] = kqsum[j0/nwarps] * KQ_max_scale + kqsum_add;

#pragma unroll
            for (int i0 = 0; i0 < D; i0 += 64) {
                VKQ[j0/nwarps][i0/64] *= KQ_max_scale;
            }
        }

        __syncthreads();

        // Load V matrix with half2 unpacking
#pragma unroll
        for (int k0 = 0; k0 < FATTN_KQ_STRIDE_TILE_F16; k0 += nwarps) {
            const int k = k0 + threadIdx.y;

#pragma unroll
            for (int i0 = 0; i0 < D; i0 += 64) {
                const int i = i0 + threadIdx.x;

                // Load half2 and branchless unpack
                const int v_idx = i / 2;
                const half2 v_h2 = V_h2[int64_t(k_VKQ_0 + k)*stride_KV2 + v_idx];
                
                // Branchless half2 unpacking
                const half v_low = __low2half(v_h2);
                const half v_high = __high2half(v_h2);
                KV_tmp[k][i] = (i & 1) ? v_high : v_low;
            }
        }

        __syncthreads();

#pragma unroll
        for (int k0 = 0; k0 < FATTN_KQ_STRIDE_TILE_F16; k0 += 1) {
            half  V_k[D/64];
            half KQ_k[ncols/nwarps];

#pragma unroll
            for (int i0 = 0; i0 < D; i0 += 64) {
                const int i = i0 + threadIdx.x;

                V_k[i0/64] = KV_tmp[k0][i];
            }
#pragma unroll
            for (int j0 = 0; j0 < ncols; j0 += nwarps) {
                const int j = j0 + threadIdx.y;

                KQ_k[j0/nwarps] = KQ[j*FATTN_KQ_STRIDE_TILE_F16 + k0];
            }

#pragma unroll
            for (int i0 = 0; i0 < D; i0 += 64) {
#pragma unroll
                for (int j0 = 0; j0 < ncols; j0 += nwarps) {
                    // Scalar half multiplication
                    VKQ[j0/nwarps][i0/64] += V_k[i0/64] * KQ_k[j0/nwarps];
                }
            }
        }

        __syncthreads();
    }

    //Attention sink: adjust running max and sum once per head
    if (sinksf && blockIdx.y == 0) {
        const half sink = __float2half(sinksf[head]);

#pragma unroll
        for (int j0 = 0; j0 < ncols; j0 += nwarps) {
            half kqmax_new_j = fmaxf(kqmax[j0/nwarps], sink);
            kqmax_new_j = warp_reduce_max<64>(kqmax_new_j);

            const half KQ_max_scale = hexp(kqmax[j0/nwarps] - kqmax_new_j);
            kqmax[j0/nwarps] = kqmax_new_j;

            const half val = hexp(sink - kqmax[j0/nwarps]);
            kqsum[j0/nwarps] = kqsum[j0/nwarps] * KQ_max_scale;
            if (threadIdx.x == 0) {
                kqsum[j0/nwarps] = kqsum[j0/nwarps] + val;  // Scalar addition
            }

#pragma unroll
            for (int i0 = 0; i0 < D; i0 += 64) {
                VKQ[j0/nwarps][i0/64] *= KQ_max_scale;
            }
        }
    }

    float2 * dst2 = (float2 *) dst;

#pragma unroll
    for (int j_VKQ_0 = 0; j_VKQ_0 < ncols; j_VKQ_0 += nwarps) {
        const int j_VKQ = j_VKQ_0 + threadIdx.y;

        if (ic0 + j_VKQ >= ne01) {
            return;
        }

        // Use scalar sum (not half2)
        half kqsum_j = kqsum[j_VKQ_0/nwarps];
        kqsum_j = warp_reduce_sum<64>((float)kqsum_j);

        const int j_dst_unrolled = ((sequence*ne01 + ic0 + j_VKQ)*ne02 + head)*gridDim.y + blockIdx.y;

#pragma unroll
        for (int i00 = 0; i00 < D; i00 += 64) {
            const int i0 = i00 + threadIdx.x;

            half dst_val = VKQ[j_VKQ_0/nwarps][i0/64];
            if (gridDim.y == 1) {
                dst_val /= kqsum_j;  // Scalar division
            }
            
            // Convert scalar half to float for output (store in appropriate position)
            const int output_idx = j_dst_unrolled*D + i0;
            reinterpret_cast<float*>(dst2)[output_idx] = __half2float(dst_val);
        }

        if (gridDim.y != 1 && threadIdx.x == 0) {
            dst_meta[j_dst_unrolled] = make_float2(kqmax[j_VKQ_0/nwarps], kqsum_j);
        }
    }
#else
    GGML_UNUSED_VARS(Q, K, V, mask, sinks, KV_max, dst, dst_meta, scale,
        max_bias, m0, m1, n_head_log2, logit_softcap,
        ne00, ne01, ne02, ne03,
              nb01, nb02, nb03,
        ne10, ne11, ne12, ne13,
              nb11, nb12, nb13,
              nb21, nb22, nb23,
              ne31, ne32, ne33,
              nb31, nb32, nb33);
    NO_DEVICE_CODE;
#endif // defined(FLASH_ATTN_AVAILABLE) && defined(FP16_AVAILABLE)
}

// GFX906 launch function for D=128 head dimension
template <int cols_per_block, bool use_logit_softcap>
void launch_fattn_tile_f16_gfx906(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    
    // Only support D=128 for GFX906 optimization
    if (Q->ne[0] != 128) {
        GGML_ABORT("GFX906 FlashAttention tile kernels only support head size 128.");
    }
    
    constexpr int    D             = 128;
    constexpr int    nwarps        = 8;
    constexpr size_t nbytes_shared = 0;
    
    // Fixed 64-thread wavefront kernel
    fattn_kernel_t fattn_kernel = flash_attn_tile_ext_f16_warp<D, cols_per_block, nwarps, use_logit_softcap>;
    
    // Launch kernel
    launch_fattn<D, cols_per_block, 1>
        (ctx, dst, fattn_kernel, nwarps, nbytes_shared, FATTN_KQ_STRIDE_TILE_F16, true, true, false, 64);
}

void ggml_cuda_flash_attn_ext_tile_f16(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * KQV = dst;
    const ggml_tensor * Q   = dst->src[0];

    const int32_t precision = KQV->op_params[3];
    GGML_ASSERT(precision == GGML_PREC_DEFAULT);

    float logit_softcap;
    memcpy(&logit_softcap, (const float *) KQV->op_params + 2, sizeof(float));

    if (Q->ne[1] <= 16) {
        constexpr int cols_per_block = 16;
        if (logit_softcap == 0.0f) {
            constexpr bool use_logit_softcap = false;
            launch_fattn_tile_f16_gfx906<cols_per_block, use_logit_softcap>(ctx, dst);
        } else {
            constexpr bool use_logit_softcap = true;
            launch_fattn_tile_f16_gfx906<cols_per_block, use_logit_softcap>(ctx, dst);
        }
        return;
    }

    constexpr int cols_per_block = 32;
    if (logit_softcap == 0.0f) {
        constexpr bool use_logit_softcap = false;
        launch_fattn_tile_f16_gfx906<cols_per_block, use_logit_softcap>(ctx, dst);
    } else {
        constexpr bool use_logit_softcap = true;
        launch_fattn_tile_f16_gfx906<cols_per_block, use_logit_softcap>(ctx, dst);
    }
}
