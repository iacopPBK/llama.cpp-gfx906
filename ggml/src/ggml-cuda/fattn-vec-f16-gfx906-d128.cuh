#include "common.cuh"
#include "fattn-common.cuh"
#include "gfx906-wave-primitives.cuh"

// GFX906 Phase A: Optimized dequantization within thread specialization pattern
// Focus: Mathematical correctness with optimized Q8_0 dequantization functions
// Preserves original thread organization - each thread handles one head dimension

// Phase A: Optimized Q8_0 dequantization for V cache
// Maintains same interface as dequantize_1_v but with GFX906 optimizations
template <int D>
__device__ __forceinline__ half dequantize_1_v_gfx906_q8_0(const char * __restrict__ V_base, int tid) {
    // Input: V_base points to the Q8_0 block for this sequence position
    // tid: thread ID representing head dimension [0, D-1]
    
    const block_q8_0 * V_q8_0 = (const block_q8_0 *) V_base;
    
    // Standard Q8_0 dequantization - preserved for mathematical correctness
    // Each thread dequantizes its own element: V[seq_pos, head_tid]
    const int block_idx = tid / QK8_0;        // Which Q8_0 block (0-3 for D=128)
    const int local_idx = tid % QK8_0;        // Index within block (0-31)
    
    if (block_idx < D/QK8_0) {
        const block_q8_0& block = V_q8_0[block_idx];
        const float d = __half2float(block.d);
        const float dequant_val = d * (float)block.qs[local_idx];
        return __float2half(dequant_val);
    }
    
    return __float2half(0.0f);  // Out of bounds
}

// Phase A: Optimized Q8_0 dequantization for K cache  
// Future optimization point for DWORDX4 operations
template <int D>
__device__ __forceinline__ half dequantize_1_k_gfx906_q8_0(const char * __restrict__ K_base, int tid) {
    // For now, use same optimization as V
    return dequantize_1_v_gfx906_q8_0<D>(K_base, tid);
}

// Currently llvm with the amdgcn target does not support unrolling loops
// that contain a break that can not be resolved at compile time.
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpass-failed"
#endif // __clang__

// GFX906-specific kernel - specialized for D=128 with Phase A optimizations
template<int ncols, ggml_type type_K, ggml_type type_V, bool use_logit_softcap>
__launch_bounds__(128, 4)  // 128 threads, 4 blocks per CU for GFX906
__global__ void flash_attn_vec_ext_f16_gfx906_d128(
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

    // GFX906 kernel is specialized for D=128 only
    constexpr int D = 128;
    static_assert(D == 128, "GFX906 kernel specialized for D=128 only");

    // Skip unused kernel variants for faster compilation:
    if (use_logit_softcap && !(D == 128 || D == 256)) {
        NO_DEVICE_CODE;
        return;
    }
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    if (ncols > 1) {
        NO_DEVICE_CODE;
        return;
    }
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

    //In this kernel Q, K, V are matrices while i, j, k are matrix indices.

    constexpr vec_dot_KQ_f16_t vec_dot_KQ = get_vec_dot_KQ_f16<D>(type_K);
    constexpr bool Q_q8_1 = type_K != GGML_TYPE_F16;
    constexpr dequantize_1_f16_t dequantize_1_v = get_dequantize_1_f16(type_V);

    const int ic0 = blockIdx.x * ncols; // Index of the Q/QKV column to work on.

    const int sequence = blockIdx.z / ne02;
    const int head = blockIdx.z - sequence*ne02;
    const int gqa_ratio = ne02 / ne12; // With grouped query attention there are > 1 Q matrices per K, V matrix.
    Q += nb03*sequence + nb02* head              + nb01*ic0;
    K += nb13*sequence + nb12*(head / gqa_ratio);
    V += nb23*sequence + nb22*(head / gqa_ratio);

    const half  * maskh  = (const half  *) (mask + nb33*(sequence % ne33) + nb31*ic0);
    const float * sinksf = (const float *) (sinks);

    const float slopef = get_alibi_slope(max_bias, head, n_head_log2, m0, m1);
    const half  slopeh = __float2half(slopef);

    static_assert(D % (2*WARP_SIZE) == 0, "D not divisible by 2*WARP_SIZE == 64.");
    constexpr int nwarps = D / WARP_SIZE;
    const int tid = WARP_SIZE*threadIdx.y + threadIdx.x;
    __builtin_assume(tid < D);

    __shared__ half KQ[ncols*D];
    half2 * KQ2 = (half2 *) KQ;

    half kqmax[ncols];
    half kqsum[ncols];
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        kqmax[j] = -HALF_MAX_HALF;
        kqsum[j] = 0.0f;
    }

    __shared__ half kqmax_shared[ncols][WARP_SIZE];
    __shared__ half kqsum_shared[ncols][WARP_SIZE];
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        if (threadIdx.y == 0) {
            kqmax_shared[j][threadIdx.x] = -HALF_MAX_HALF;
            kqsum_shared[j][threadIdx.x] = 0.0f;
        }
    }

    __shared__ half maskh_shared[ncols*D];
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        maskh_shared[j*D + tid] = 0.0f;
    }

    __syncthreads();

    // Convert Q to half2 (f16 K) or q8_1 (quantized K) and store in registers:
    half2  Q_h2[ncols][D/(2*WARP_SIZE)];
    int   Q_i32[ncols][D/(sizeof(int)*QK8_1) == 0 ? 1 : D/(sizeof(int)*QK8_1)];
    half2  Q_ds[ncols][D/QK8_1 == 0 ? 1 : D/QK8_1];
    if (Q_q8_1) {
#pragma unroll
        for (int j0 = 0; j0 < ncols; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

            if (j0 + nwarps > ncols && j >= ncols) {
                break;
            }

            // Reuse KQ as temporary storage for converting Q to q8_1:
            int   * tmp_q_i32 = (int   *) &KQ[j*D];
            half2 * tmp_q_ds  = (half2 *) (tmp_q_i32 + D/sizeof(int));

            // Set memory to zero if out of bounds:
            if (ncols > 2 && ic0 + j >= ne01) {
#pragma unroll
                for (int i0 = 0; i0 < D/sizeof(int); i0 += WARP_SIZE) {
                    const int i = i0 + threadIdx.x;

                    tmp_q_i32[i] = 0;
                }
                if (threadIdx.x < D/QK8_1) {
                    tmp_q_ds[threadIdx.x] = make_half2(0.0f, 0.0f);
                }
                continue;
            }

            const float * Q_f = (const float *) (Q + j*nb01);
#pragma unroll
            for (int i0 = 0; i0 < D/sizeof(int); i0 += WARP_SIZE) {
                quantize_q8_1_to_shared<half2>(Q_f + 4*i0, scale, tmp_q_i32, tmp_q_ds);
            }
        }

        __syncthreads();

#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            int   * tmp_q_i32 = (int   *) &KQ[j*D];
            half2 * tmp_q_ds  = (half2 *) (tmp_q_i32 + D/sizeof(int));

#pragma unroll
            for (int i0 = 0; i0 < D/sizeof(int); i0 += WARP_SIZE) {
                const int i = i0 + threadIdx.x;

                Q_i32[j][i0/WARP_SIZE] = tmp_q_i32[i];
                Q_ds[j][i0/WARP_SIZE]  = tmp_q_ds[i/QI8_1];
            }
        }

        __syncthreads();
    } else {
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            const float2 * Q_f2_j = (const float2 *) (Q + j*nb01);

#pragma unroll
            for (int i0 = 0; i0 < D/2; i0 += WARP_SIZE) {
                const int i = i0 + threadIdx.x;

                const float2 tmp = ncols <= 2 || ic0 + j < ne01 ? Q_f2_j[i] : make_float2(0.0f, 0.0f);
                Q_h2[j][i0/WARP_SIZE] = make_half2(scale, scale) * make_half2(tmp.x, tmp.y);
            }
        }
    }

#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        KQ[j*D + tid] = -HALF_MAX_HALF;
    }
    __syncthreads();

    half2 VKQ[ncols] = {{0.0f, 0.0f}};

    const int k_VKQ_max = KV_max ? KV_max[sequence*gridDim.x + blockIdx.x] : ne11;
    K     += blockIdx.y*D * nb11;
    V     += blockIdx.y*D * nb21;
    maskh += blockIdx.y*D;
    for (int k_VKQ_0 = blockIdx.y*D; k_VKQ_0 < k_VKQ_max; k_VKQ_0 += gridDim.y*D,
             // Increment pointers after each loop:
             K += gridDim.y*D*nb11, V += gridDim.y*D*nb21, maskh += gridDim.y*D) {

        // Calculate KQ tile and keep track of new maximum KQ values:

        if (mask) {
#pragma unroll
            for (int j = 0; j < ncols; ++j) {
                maskh_shared[j*D + tid] = slopeh*maskh[j*ne11 + tid];
            }
            __syncthreads();
        }

        // For unknown reasons using a half array of size 1 for kqmax_new causes a performance regression,
        // see https://github.com/ggerganov/llama.cpp/pull/7061 .
        // Therefore this variable is defined twice but only used once (so that the compiler can optimize out the unused variable).
        half kqmax_new = kqmax[0];
        half kqmax_new_arr[ncols];
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            kqmax_new_arr[j] = kqmax[j];
        }

#pragma unroll
        for (int i_KQ_0 = 0; i_KQ_0 < D; i_KQ_0 += nwarps) {
            const int i_KQ = i_KQ_0 + threadIdx.y;

            if ((i_KQ_0 + nwarps > D && i_KQ >= D) || (FATTN_KQ_STRIDE % D != 0 && k_VKQ_0 + i_KQ >= ne11)) {
                break;
            }

#pragma unroll
            for (int j = 0; j < ncols; ++j) {
                half sum = vec_dot_KQ(K + i_KQ*nb11, Q_h2[j], Q_i32[j], Q_ds[j]);
                sum = warp_reduce_sum((float)sum);

                if (use_logit_softcap) {
                    sum = logit_softcap*tanhf(sum);
                }

                sum += maskh_shared[j*D + i_KQ];

                if (ncols == 1) {
                    kqmax_new        = ggml_cuda_hmax(kqmax_new,        sum);
                } else {
                    kqmax_new_arr[j] = ggml_cuda_hmax(kqmax_new_arr[j], sum);
                }

                if (threadIdx.x == 0) {
                    KQ[j*D + i_KQ] = sum;
                }
            }
        }

#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            half kqmax_new_j = ncols == 1 ? kqmax_new : kqmax_new_arr[j];

            if (threadIdx.x == 0) {
                kqmax_shared[j][threadIdx.y] = kqmax_new_j;
            }
        }

        __syncthreads();

#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            half kqmax_new_j = kqmax_shared[j][threadIdx.x];
            kqmax_new_j = warp_reduce_max(kqmax_new_j);

            const half KQ_max_scale = hexp(kqmax[j] - kqmax_new_j);
            kqmax[j] = kqmax_new_j;

            const half val = hexp(KQ[j*D + tid] - kqmax[j]);
            kqsum[j] = kqsum[j]*KQ_max_scale + val;
            KQ[j*D + tid] = val;

            VKQ[j] *= __half2half2(KQ_max_scale);
        }

        __syncthreads();

        // Phase A: Optimized V processing with thread specialization preserved
        // Each thread tid handles head dimension tid across all sequence positions
#pragma unroll
        for (int k0 = 0; k0 < D; k0 += 2) {
            if (FATTN_KQ_STRIDE % D != 0 && k_VKQ_0 + k0 >= ne11) {
                break;
            }

            half2 V_k;
            
            // Phase A optimization: Use GFX906-optimized Q8_0 dequantization when applicable
            if constexpr (type_V == GGML_TYPE_Q8_0) {
                // Thread tid dequantizes V[seq_k0, head_tid] and V[seq_k0+1, head_tid]
                reinterpret_cast<half&>(V_k.x) = dequantize_1_v_gfx906_q8_0<D>(V + (k0 + 0)*nb21, tid);
                reinterpret_cast<half&>(V_k.y) = dequantize_1_v_gfx906_q8_0<D>(V + (k0 + 1)*nb21, tid);
            } else {
                // Standard dequantization for other V types
                reinterpret_cast<half&>(V_k.x) = dequantize_1_v(V + (k0 + 0)*nb21, tid);
                reinterpret_cast<half&>(V_k.y) = dequantize_1_v(V + (k0 + 1)*nb21, tid);
            }

            // Standard Flash Attention accumulation - unchanged
#pragma unroll
            for (int j = 0; j < ncols; ++j) {
                VKQ[j] += V_k*KQ2[j*(D/2) + k0/2];
            }
        }

        __syncthreads();
    }

    if (sinksf && blockIdx.y == 0) {
        const half sink = __float2half(sinksf[head]);

#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            if (threadIdx.x == 0) {
                kqmax_shared[j][threadIdx.y] = fmaxf(kqmax[j], sink);
            }
        }

        __syncthreads();

#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            half kqmax_new_j = kqmax_shared[j][threadIdx.x];
            kqmax_new_j = warp_reduce_max(kqmax_new_j);

            const half KQ_max_scale = hexp(kqmax[j] - kqmax_new_j);
            kqmax[j] = kqmax_new_j;

            const half val = hexp(sink - kqmax[j]);
            kqsum[j] = kqsum[j]*KQ_max_scale;

            if (tid == 0) {
                kqsum[j] += val;
            }

            VKQ[j] *= __half2half2(KQ_max_scale);
        }

        __syncthreads();
    }

#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        kqsum[j] = __float2half(warp_reduce_sum(__half2float(kqsum[j])));
        if (threadIdx.x == 0) {
            kqsum_shared[j][threadIdx.y] = kqsum[j];
        }
    }

    __syncthreads();

#pragma unroll
    for (int j_VKQ = 0; j_VKQ < ncols; ++j_VKQ) {
        if (ncols > 2 && ic0 + j_VKQ >= ne01) {
            break;
        }

        kqsum[j_VKQ] = kqsum_shared[j_VKQ][threadIdx.x];
        kqsum[j_VKQ] = __float2half(warp_reduce_sum(__half2float(kqsum[j_VKQ])));

        half dst_val = (__low2half(VKQ[j_VKQ]) + __high2half(VKQ[j_VKQ]));
        if (gridDim.y == 1) {
            dst_val /= kqsum[j_VKQ];
        }
        dst[(((sequence*ne01 + ic0 + j_VKQ)*ne02 + head)*gridDim.y + blockIdx.y)*D + tid] = dst_val;
    }

    if (gridDim.y != 1 && tid < ncols && (ncols <= 2 || ic0 + tid < ne01)) {
        dst_meta[((sequence*ne01 + ic0 + tid)*ne02 + head)*gridDim.y + blockIdx.y] = make_float2(kqmax[tid], kqsum[tid]);
    }
#else
    GGML_UNUSED_VARS(Q, K, V, mask, sinks, KV_max, dst, dst_meta, scale, max_bias, m0, m1, n_head_log2, logit_softcap);
    GGML_UNUSED_VARS(ne00, ne01, ne02, ne03, nb01, nb02, nb03, ne10, ne11, ne12, ne13, nb11, nb12, nb13);
    GGML_UNUSED_VARS(nb21, nb22, nb23, ne31, ne32, ne33, nb31, nb32, nb33);
    NO_DEVICE_CODE;
#endif // defined(FLASH_ATTN_AVAILABLE) && defined(FP16_AVAILABLE)
}

#ifdef __clang__
#pragma clang diagnostic pop
#endif // __clang__

// GFX906 dispatch function for D=128 kernels
template <ggml_type type_K, ggml_type type_V>
void ggml_cuda_flash_attn_ext_vec_f16_gfx906_d128_case(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * KQV = dst;
    const ggml_tensor * Q   = dst->src[0];
    const ggml_tensor * K   = dst->src[1];
    const ggml_tensor * V   = dst->src[2];

    const int32_t precision = KQV->op_params[3];
    GGML_ASSERT(precision == GGML_PREC_DEFAULT);

    GGML_ASSERT(K->type == type_K);
    GGML_ASSERT(V->type == type_V);
    GGML_ASSERT(Q->ne[0] == 128); // GFX906 kernel specialized for D=128

    float logit_softcap;
    memcpy(&logit_softcap, (const float *) KQV->op_params + 2, sizeof(float));

    // GFX906-specific dispatch - always use our optimized kernel
    constexpr int D = 128;
    constexpr int cols_per_block = 1; // GFX906 optimized for single column
    constexpr int nwarps = 4; // 128 threads = 4 warps

    const bool use_logit_softcap = logit_softcap != 0.0f;

    // Use proper fattn_kernel_t type and standard parameter calculation
    fattn_kernel_t fattn_kernel;
    if (use_logit_softcap) {
        fattn_kernel = flash_attn_vec_ext_f16_gfx906_d128<cols_per_block, type_K, type_V, true>;
    } else {
        fattn_kernel = flash_attn_vec_ext_f16_gfx906_d128<cols_per_block, type_K, type_V, false>;
    }

    // Standard parameters matching original kernel dispatch
    constexpr bool need_f16_K = D != 128;  // GFX906 optimized for D=128, so false
    constexpr bool need_f16_V = D != 128 && D != 64;  // false for D=128
    constexpr size_t nbytes_shared = 0;

    launch_fattn<D, cols_per_block, 1>(ctx, dst, fattn_kernel, nwarps, nbytes_shared, D, need_f16_K, need_f16_V, false);
}

// Template instantiation for GFX906 kernel
template __global__ void flash_attn_vec_ext_f16_gfx906_d128<1, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, false>(
    const char * Q, const char * K, const char * V, const char * mask, const char * sinks, const int * KV_max, float * dst, float2 * dst_meta,
    const float scale, const float max_bias, const float m0, const float m1, const uint32_t n_head_log2, const float logit_softcap,
    const int32_t ne00, const int32_t ne01, const int32_t ne02, const int32_t ne03, const int32_t nb01, const int32_t nb02, const int32_t nb03,
    const int32_t ne10, const int32_t ne11, const int32_t ne12, const int32_t ne13, const int32_t nb11, const int32_t nb12, const int64_t nb13,
    const int32_t nb21, const int32_t nb22, const int64_t nb23, const int32_t ne31, const int32_t ne32, const int32_t ne33,
    const int32_t nb31, const int32_t nb32, const int64_t nb33);

template __global__ void flash_attn_vec_ext_f16_gfx906_d128<1, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, true>(
    const char * Q, const char * K, const char * V, const char * mask, const char * sinks, const int * KV_max, float * dst, float2 * dst_meta,
    const float scale, const float max_bias, const float m0, const float m1, const uint32_t n_head_log2, const float logit_softcap,
    const int32_t ne00, const int32_t ne01, const int32_t ne02, const int32_t ne03, const int32_t nb01, const int32_t nb02, const int32_t nb03,
    const int32_t ne10, const int32_t ne11, const int32_t ne12, const int32_t ne13, const int32_t nb11, const int32_t nb12, const int64_t nb13,
    const int32_t nb21, const int32_t nb22, const int64_t nb23, const int32_t ne31, const int32_t ne32, const int32_t ne33,
    const int32_t nb31, const int32_t nb32, const int64_t nb33);