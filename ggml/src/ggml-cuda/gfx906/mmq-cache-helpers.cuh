// gfx906/mmq-cache-helpers.cuh
// Helper functions for MMQ Q8 cache integration in mmq.cu

#pragma once

#include "../common.cuh"
#include "gfx906-context.cuh"

#if defined(GGML_USE_HIP) && GFX906_KVQ_MOE_CACHE_ENABLED

#include "fused/gather-q8.cuh"

// ============================================================================
// Q8 Cache Helper for Regular MUL_MAT Path
// ============================================================================

// Returns true if cache hit, false if miss (caller must quantize)
static inline bool gfx906_mmq_try_q8_cache(
        ggml_backend_cuda_context& ctx,
        const float* src1_d,
        const ggml_tensor* src1,
        ggml_type src0_type,
        int64_t ne10, int64_t ne11, int64_t ne12, int64_t ne13,
        int64_t ne10_padded,
        const char** out_q8_ptr,
        size_t* out_nbytes,
        cudaStream_t stream) {
    
    const int layout = static_cast<int>(mmq_get_q8_1_ds_layout(src0_type));
    
    const q8_cache_entry* cached = GFX906_Q8_CACHE(ctx).lookup(
        src1, layout, ne10_padded, ne11, ne12, ne13);
    
    if (cached) {
        *out_q8_ptr = static_cast<const char*>(cached->q8_data);
        return true;
    }
    
    // Cache miss - need to quantize
    const size_t nbytes = ne13*ne12 * ne11*ne10_padded * sizeof(block_q8_1)/QK8_1 +
        get_mmq_x_max_host(ggml_cuda_info().devices[ggml_cuda_get_device()].cc)*sizeof(block_q8_1_mmq);
    
    void* q8_data = GFX906_Q8_CACHE(ctx).get_buffer(nbytes);
    
    const size_t ts_src1 = ggml_type_size(src1->type);
    const int64_t s11 = src1->nb[1] / ts_src1;
    const int64_t s12 = src1->nb[2] / ts_src1;
    const int64_t s13 = src1->nb[3] / ts_src1;
    
    quantize_mmq_q8_1_cuda(src1_d, nullptr, static_cast<char*>(q8_data), src0_type,
                           ne10, s11, s12, s13, ne10_padded, ne11, ne12, ne13, stream);
    CUDA_CHECK(cudaGetLastError());
    
    GFX906_Q8_CACHE(ctx).store(src1, layout, q8_data, nbytes,
                       ne10_padded, ne11, ne12, ne13);
    
    *out_q8_ptr = static_cast<const char*>(q8_data);
    *out_nbytes = nbytes;
    return true;
}

// ============================================================================
// MoE Cache Helper for MUL_MAT_ID Path
// ============================================================================

// Returns true if cache was used (gather done), false if caller should quantize
static inline bool gfx906_mmq_try_moe_cache(
        ggml_backend_cuda_context& ctx,
        const float* src1_d,
        const ggml_tensor* src1,
        ggml_type src0_type,
        int64_t ne10, int64_t ne11, int64_t ne12, int64_t ne13,
        int64_t ne10_padded,
        int64_t ne11_flat,
        const int32_t* ids_src1,
        char* src1_q8_1,
        cudaStream_t stream) {
    
    const int layout = static_cast<int>(mmq_get_q8_1_ds_layout(src0_type));
    
    const q8_cache_entry* moe_cached = GFX906_Q8_CACHE(ctx).lookup(
        src1, layout, ne10_padded, ne11, ne12, ne13);
    
    if (!moe_cached) {
        const size_t full_nbytes = ne13*ne12*ne11*ne10_padded * sizeof(block_q8_1)/QK8_1 +
            get_mmq_x_max_host(ggml_cuda_info().devices[ggml_cuda_get_device()].cc)*sizeof(block_q8_1_mmq);
        
        void* full_q8 = GFX906_Q8_CACHE(ctx).get_buffer(full_nbytes);
        
        const size_t ts_src1 = ggml_type_size(src1->type);
        const int64_t s11 = src1->nb[1] / ts_src1;
        const int64_t s12 = src1->nb[2] / ts_src1;
        const int64_t s13 = src1->nb[3] / ts_src1;
        
        quantize_mmq_q8_1_cuda(src1_d, nullptr, static_cast<char*>(full_q8), src0_type,
                               ne10, s11, s12, s13, ne10_padded,
                               ne11, ne12, ne13, stream);
        CUDA_CHECK(cudaGetLastError());
        
        GFX906_Q8_CACHE(ctx).store(src1, layout, full_q8, full_nbytes,
                           ne10_padded, ne11, ne12, ne13);
        
        moe_cached = GFX906_Q8_CACHE(ctx).lookup(src1, layout, ne10_padded, ne11, ne12, ne13);
    }
    
    // Gather selected rows from cached full Q8_1 tensor
    const int64_t block_size = sizeof(block_q8_1_mmq);
    const int64_t n_blocks = ne10_padded / (4*QK8_1);
    
    gather_q8_1_rows_cuda(
        moe_cached->q8_data,
        ids_src1,
        src1_q8_1,
        block_size,
        n_blocks,
        ne11,
        ne11_flat,
        stream
    );
    CUDA_CHECK(cudaGetLastError());
    
    return true;
}

#else

// No-op versions for non-HIP builds

static inline bool gfx906_mmq_try_q8_cache(
        ggml_backend_cuda_context& ctx,
        const float* src1_d,
        const ggml_tensor* src1,
        ggml_type src0_type,
        int64_t ne10, int64_t ne11, int64_t ne12, int64_t ne13,
        int64_t ne10_padded,
        const char** out_q8_ptr,
        size_t* out_nbytes,
        cudaStream_t stream) {
    (void)ctx; (void)src1_d; (void)src1; (void)src0_type;
    (void)ne10; (void)ne11; (void)ne12; (void)ne13; (void)ne10_padded;
    (void)out_q8_ptr; (void)out_nbytes; (void)stream;
    return false;
}

static inline bool gfx906_mmq_try_moe_cache(
        ggml_backend_cuda_context& ctx,
        const float* src1_d,
        const ggml_tensor* src1,
        ggml_type src0_type,
        int64_t ne10, int64_t ne11, int64_t ne12, int64_t ne13,
        int64_t ne10_padded,
        int64_t ne11_flat,
        const int32_t* ids_src1,
        char* src1_q8_1,
        cudaStream_t stream) {
    (void)ctx; (void)src1_d; (void)src1; (void)src0_type;
    (void)ne10; (void)ne11; (void)ne12; (void)ne13; (void)ne10_padded;
    (void)ne11_flat; (void)ids_src1; (void)src1_q8_1; (void)stream;
    return false;
}

#endif // defined(GGML_USE_HIP) && GFX906_KVQ_MOE_CACHE_ENABLED
