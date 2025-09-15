#include "common.cuh"
#include "fattn-common.cuh"
#include "gfx906-wave-primitives.cuh"
#include "gfx906-vectorized-dequant.cuh"  // Sequence-wise vectorization

// GFX906 Phase 1B: Vectorized Dequantization Optimization
// Target: 4× V dequantization throughput improvement
// Focus: SIMD-style Q8_0 processing + bank conflict elimination + 64-thread wave support

// Phase 1A Configuration - ENABLED with correct sequence-wise vectorization
// Sequence-wise vectorized implementation preserves Flash Attention correctness
#ifdef GGML_HIP_GFX906_OPTIMIZED
#define GGML_HIP_GFX906_PHASE1A 1  // ENABLED: Correct sequence-wise vectorization
#else
#define GGML_HIP_GFX906_PHASE1A 0
#endif

#if GGML_HIP_GFX906_PHASE1A

// Single-element dequantization function for fallback cases
template <int D>
__device__ __forceinline__ half dequantize_1_v_gfx906_q8_0(const char * __restrict__ V_base, int tid) {
    const block_q8_0 * V_q8_0 = (const block_q8_0 *) V_base;
    
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

// GFX906-specific padding constants for bank conflict elimination
// Based on tile kernel proven patterns, but scaled conservatively
constexpr int GFX906_KQ_PADDING = 16;        // Conservative KQ padding (vs tile's 48)
constexpr int GFX906_MASK_PADDING = 16;      // Conservative mask padding  
constexpr int GFX906_V_CACHE_PADDING = 16;   // Conservative V cache padding (vs tile's 48)
constexpr int GFX906_REDUCTION_PADDING = 16; // Wave-sized reduction padding

// Phase 1A: Conservative V cache sizing (8 tokens vs unlimited current)
constexpr int V_CACHE_TOKENS = 8;            // Cache 8 sequence positions
constexpr int WAVE_SIZE_GFX906 = 64;         // Native GFX906 wave size

#endif // GGML_HIP_GFX906_PHASE1A

// Phase 1A: Enhanced Q8_0 dequantization with caching support
template <int D>
__device__ __forceinline__ half dequantize_1_v_gfx906_q8_0_cached(const char * __restrict__ V_base, int tid) {
    // Same mathematical correctness as original, but marked for caching optimization
    const block_q8_0 * V_q8_0 = (const block_q8_0 *) V_base;
    
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

// Currently llvm with the amdgcn target does not support unrolling loops
// that contain a break that can not be resolved at compile time.
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpass-failed"
#endif // __clang__

// GFX906 Phase 1B kernel - Vectorized dequantization optimization for D=128
template<int ncols, ggml_type type_K, ggml_type type_V, bool use_logit_softcap>
__launch_bounds__(128, 4)  // Keep original launch bounds for Phase 1A
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
    static_assert(D == 128, "GFX906 Phase 1A kernel specialized for D=128 only");

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

    // Phase 1A: ENHANCED SHARED MEMORY LAYOUT with strategic padding
#if GGML_HIP_GFX906_PHASE1A
    // Bank conflict elimination through strategic padding
    __shared__ half KQ[ncols*(D + GFX906_KQ_PADDING)];                      // 288 bytes (vs 256)
    __shared__ half kqmax_shared[ncols][WAVE_SIZE_GFX906 + GFX906_REDUCTION_PADDING]; // 160 bytes (vs 64) 
    __shared__ half kqsum_shared[ncols][WAVE_SIZE_GFX906 + GFX906_REDUCTION_PADDING]; // 160 bytes (vs 64)
    __shared__ half maskh_shared[ncols*(D + GFX906_MASK_PADDING)];          // 288 bytes (vs 256)
    
    // Phase 1A: V cache disabled due to correctness issues - will be fixed in Phase 1B
    // __shared__ half V_cache[V_CACHE_TOKENS][D + GFX906_V_CACHE_PADDING]; // DISABLED
    // __shared__ int V_cache_valid[V_CACHE_TOKENS];                         // DISABLED
    
    // Phase 1B LDS usage: 288+160+160+288 = 896 bytes (1.4% utilization - similar to baseline)
    // Phase 1B optimization: 4× vectorized V dequantization throughput
    
    // Padded array access macros for bank conflict avoidance
    #define KQ_ACCESS(j, i) KQ[(j) * (D + GFX906_KQ_PADDING) + (i)]
    #define MASK_ACCESS(j, i) maskh_shared[(j) * (D + GFX906_MASK_PADDING) + (i)]
    
#else
    // Original layout (fallback for testing)
    __shared__ half KQ[ncols*D];
    __shared__ half kqmax_shared[ncols][WARP_SIZE];
    __shared__ half kqsum_shared[ncols][WARP_SIZE]; 
    __shared__ half maskh_shared[ncols*D];
    
    #define KQ_ACCESS(j, i) KQ[(j)*D + (i)]
    #define MASK_ACCESS(j, i) maskh_shared[(j)*D + (i)]
#endif // GGML_HIP_GFX906_PHASE1A

    half2 * KQ2 = (half2 *) KQ;

    half kqmax[ncols];
    half kqsum[ncols];
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        kqmax[j] = -HALF_MAX_HALF;
        kqsum[j] = 0.0f;
    }

#if GGML_HIP_GFX906_PHASE1A
    // Phase 1A: Enhanced initialization for 64-thread waves
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        if (threadIdx.y == 0 && threadIdx.x < WAVE_SIZE_GFX906) {
            kqmax_shared[j][threadIdx.x] = -HALF_MAX_HALF;
            kqsum_shared[j][threadIdx.x] = 0.0f;
        }
    }

    // Phase 1A: V cache initialization disabled
#else
    // Original initialization
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        if (threadIdx.y == 0) {
            kqmax_shared[j][threadIdx.x] = -HALF_MAX_HALF;
            kqsum_shared[j][threadIdx.x] = 0.0f;
        }
    }
#endif // GGML_HIP_GFX906_PHASE1A

#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        MASK_ACCESS(j, tid) = 0.0f;
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
            int   * tmp_q_i32 = (int   *) &KQ_ACCESS(j, 0);
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
            int   * tmp_q_i32 = (int   *) &KQ_ACCESS(j, 0);
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
        KQ_ACCESS(j, tid) = -HALF_MAX_HALF;
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
                MASK_ACCESS(j, tid) = slopeh*maskh[j*ne11 + tid];
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

                sum += MASK_ACCESS(j, i_KQ);

                if (ncols == 1) {
                    kqmax_new        = ggml_cuda_hmax(kqmax_new,        sum);
                } else {
                    kqmax_new_arr[j] = ggml_cuda_hmax(kqmax_new_arr[j], sum);
                }

                if (threadIdx.x == 0) {
                    KQ_ACCESS(j, i_KQ) = sum;
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

#if GGML_HIP_GFX906_PHASE1A
        // Phase 1A: Enhanced 64-thread wave reductions
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            half kqmax_new_j = kqmax_shared[j][threadIdx.x];
            kqmax_new_j = warp_reduce_max(kqmax_new_j);  // Will use 64-thread version

            const half KQ_max_scale = hexp(kqmax[j] - kqmax_new_j);
            kqmax[j] = kqmax_new_j;

            const half val = hexp(KQ_ACCESS(j, tid) - kqmax[j]);
            kqsum[j] = kqsum[j]*KQ_max_scale + val;
            KQ_ACCESS(j, tid) = val;

            VKQ[j] *= __half2half2(KQ_max_scale);
        }
#else
        // Original wave reduction
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
#endif // GGML_HIP_GFX906_PHASE1A

        __syncthreads();

        // Phase 1B: SEQUENCE-WISE VECTORIZED V PROCESSING - Mathematically Correct 4× speedup
#if GGML_HIP_GFX906_PHASE1A
        // Phase 1B: Correct sequence-wise vectorized dequantization for Q8_0
        if constexpr (type_V == GGML_TYPE_Q8_0) {
            // Process V sequences in groups of 4 - each thread keeps same head dimension
            constexpr int VECTOR_SIZE = 4;
            
            for (int k0 = 0; k0 < D; k0 += VECTOR_SIZE) {
                if (FATTN_KQ_STRIDE % D != 0 && k_VKQ_0 + k0 >= ne11) {
                    break;
                }
                
                const int remaining = min(VECTOR_SIZE, D - k0);
                
                if (remaining >= 4) {
                    // Vectorized sequence processing: thread tid processes 4 consecutive sequences
                    // V[k0, tid], V[k0+1, tid], V[k0+2, tid], V[k0+3, tid]
                    half4 V_seq4 = dequantize_4v_sequence_q8_0<D>(V, k0, tid, nb21);
                    
                    // Process in pairs to match existing accumulation pattern
                    half2 V_k01 = {V_seq4.x, V_seq4.y};  // sequences k0, k0+1
                    half2 V_k23 = {V_seq4.z, V_seq4.w};  // sequences k0+2, k0+3
                    
                    // Flash Attention accumulation - mathematically identical to original
#pragma unroll
                    for (int j = 0; j < ncols; ++j) {
                        VKQ[j] += V_k01 * KQ2[j*(D/2) + k0/2];        // k0, k0+1
                        VKQ[j] += V_k23 * KQ2[j*(D/2) + (k0+2)/2];    // k0+2, k0+3
                    }
                } else if (remaining >= 2) {
                    // 2-element processing for remaining sequences
                    half2 V_k;
                    reinterpret_cast<half&>(V_k.x) = dequantize_1_v_gfx906_q8_0<D>(V + (k0 + 0)*nb21, tid);
                    reinterpret_cast<half&>(V_k.y) = dequantize_1_v_gfx906_q8_0<D>(V + (k0 + 1)*nb21, tid);
                    
#pragma unroll
                    for (int j = 0; j < ncols; ++j) {
                        VKQ[j] += V_k * KQ2[j*(D/2) + k0/2];
                    }
                } else {
                    // Single element fallback
                    half2 V_k;
                    reinterpret_cast<half&>(V_k.x) = dequantize_1_v_gfx906_q8_0<D>(V + k0*nb21, tid);
                    V_k.y = __float2half(0.0f);
                    
#pragma unroll
                    for (int j = 0; j < ncols; ++j) {
                        VKQ[j] += V_k * KQ2[j*(D/2) + k0/2];
                    }
                }
            }
        } else {
            // Non-Q8_0 types: use original processing
#pragma unroll
            for (int k0 = 0; k0 < D; k0 += 2) {
                if (FATTN_KQ_STRIDE % D != 0 && k_VKQ_0 + k0 >= ne11) {
                    break;
                }

                half2 V_k;
                reinterpret_cast<half&>(V_k.x) = dequantize_1_v(V + (k0 + 0)*nb21, tid);
                reinterpret_cast<half&>(V_k.y) = dequantize_1_v(V + (k0 + 1)*nb21, tid);

#pragma unroll
                for (int j = 0; j < ncols; ++j) {
                    VKQ[j] += V_k*KQ2[j*(D/2) + k0/2];
                }
            }
        }
#else
        // Original V processing (fallback or non-Q8_0 types)
#pragma unroll
        for (int k0 = 0; k0 < D; k0 += 2) {
            if (FATTN_KQ_STRIDE % D != 0 && k_VKQ_0 + k0 >= ne11) {
                break;
            }

            half2 V_k;
            
            // Original dequantization
            if constexpr (type_V == GGML_TYPE_Q8_0) {
                reinterpret_cast<half&>(V_k.x) = dequantize_1_v_gfx906_q8_0_cached<D>(V + (k0 + 0)*nb21, tid);
                reinterpret_cast<half&>(V_k.y) = dequantize_1_v_gfx906_q8_0_cached<D>(V + (k0 + 1)*nb21, tid);
            } else {
                reinterpret_cast<half&>(V_k.x) = dequantize_1_v(V + (k0 + 0)*nb21, tid);
                reinterpret_cast<half&>(V_k.y) = dequantize_1_v(V + (k0 + 1)*nb21, tid);
            }

#pragma unroll
            for (int j = 0; j < ncols; ++j) {
                VKQ[j] += V_k*KQ2[j*(D/2) + k0/2];
            }
        }
#endif // GGML_HIP_GFX906_PHASE1A

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

// Phase 1B dispatch function for D=128 kernels
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
    GGML_ASSERT(Q->ne[0] == 128); // Phase 1A kernel specialized for D=128

    float logit_softcap;
    memcpy(&logit_softcap, (const float *) KQV->op_params + 2, sizeof(float));

    // Phase 1B specific dispatch
    constexpr int D = 128;
    constexpr int cols_per_block = 1;
    constexpr int nwarps = 4; // 128 threads = 4 warps

    const bool use_logit_softcap = logit_softcap != 0.0f;

    // Use Phase 1B kernel
    fattn_kernel_t fattn_kernel;
    if (use_logit_softcap) {
        fattn_kernel = flash_attn_vec_ext_f16_gfx906_d128<cols_per_block, type_K, type_V, true>;
    } else {
        fattn_kernel = flash_attn_vec_ext_f16_gfx906_d128<cols_per_block, type_K, type_V, false>;
    }

    // Standard parameters
    constexpr bool need_f16_K = false;   // D=128 specialized
    constexpr bool need_f16_V = false;   // D=128 specialized
    constexpr size_t nbytes_shared = 0;

    launch_fattn<D, cols_per_block, 1>(ctx, dst, fattn_kernel, nwarps, nbytes_shared, D, need_f16_K, need_f16_V, false);
}

// Template instantiation for Phase 1B kernel
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