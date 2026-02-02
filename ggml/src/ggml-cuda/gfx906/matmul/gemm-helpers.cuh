#pragma once

// GFX906 GEMM Helper Functions
// Provides dispatch to custom GEMM implementations for AMD GPUs

#include "../gfx906-config.h"

#if defined(GGML_USE_HIP)

#include "mmf.cuh"

// Forward declaration for sgemm dispatch (if available)
bool gfx906_sgemm_dispatch(
    const float * src0, const float * src1, float * dst,
    int m, int n, int k,
    int lda, int ldb, int ldc,
    hipStream_t stream);

// ============================================================================
// FP16 GEMM Dispatch
// ============================================================================
// Attempts to use GFX906 custom FP16 GEMM for medium batch sizes.
// Returns true if handled, false to fall back to cublasGemmEx.

static inline bool gfx906_fp16_gemm_dispatch(
        const void * src0_ptr,
        const void * src1_ptr,
        float * dst_dd_i,
        int row_diff,
        int src1_ncols,
        int ne10,
        int ne00,
        int ldc,
        hipStream_t stream,
        int cc) {
    
    // Only use custom GEMM on GCN architecture (GFX906)
    if (!GGML_CUDA_CC_IS_GCN(cc)) {
        return false;
    }
    
    return gfx906_mmf_dispatch(
        (const half *)src0_ptr, (const half *)src1_ptr, dst_dd_i,
        row_diff, src1_ncols, ne10,
        ne00, ne10, ldc,
        stream
    );
}

// ============================================================================
// SGEMM Dispatch
// ============================================================================
// Attempts to use GFX906 custom SGEMM for medium batch sizes.
// Returns true if handled, false to fall back to cublasSgemm.

static inline bool gfx906_sgemm_dispatch_wrapper(
        const float * src0_ddf_i,
        const float * src1_ddf1_i,
        float * dst_dd_i,
        int row_diff,
        int src1_ncols,
        int ne10,
        int ne00,
        int ldc,
        hipStream_t stream,
        int cc) {
    
    // Only use custom GEMM on GCN architecture (GFX906)
    if (!GGML_CUDA_CC_IS_GCN(cc)) {
        return false;
    }
    
    return gfx906_sgemm_dispatch(
        src0_ddf_i, src1_ddf1_i, dst_dd_i,
        row_diff, src1_ncols, ne10,
        ne00, ne10, ldc,
        stream
    );
}

#endif // GGML_USE_HIP
