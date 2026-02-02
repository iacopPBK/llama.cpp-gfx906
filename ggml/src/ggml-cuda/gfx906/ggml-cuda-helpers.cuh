#pragma once

// GFX906 ggml-cuda.cu Helper Functions
// Contains hooks for graph evaluation, context cleanup, and fusion

#include "gfx906-config.h"

#if defined(GGML_USE_HIP)

// Include fusion headers only when Q8 cache is enabled
#if GFX906_KVQ_MOE_CACHE_ENABLED
#include "fused/graph-fusion.cuh"
#endif

// ============================================================================
// Context Destructor Cleanup
// ============================================================================

#if GFX906_KVQ_MOE_CACHE_ENABLED
static inline void gfx906_context_cleanup(ggml_backend_cuda_context * ctx) {
    GFX906_Q8_CACHE(*ctx).free_all();
    GFX906_FREE_CONTEXT(*ctx);
}
#endif

// ============================================================================
// Graph Evaluation Setup
// ============================================================================

#if GFX906_KVQ_MOE_CACHE_ENABLED
static inline void gfx906_graph_eval_setup(
        ggml_backend_cuda_context * cuda_ctx,
        bool use_cuda_graph) {
    GFX906_CLEAR_Q8_CACHE(*cuda_ctx);
    
    // Sync before clearing fusion buffers (skip during graph capture)
    if (!GFX906_Q8_BUFFERS(*cuda_ctx).empty() && !use_cuda_graph) {
        CUDA_CHECK(cudaStreamSynchronize(cuda_ctx->stream()));
    }
    clear_fusion_state(cuda_ctx);
}

// ============================================================================
// Fusion Hooks
// ============================================================================

static inline bool gfx906_try_rms_mul_mmq_fusion(
        ggml_backend_cuda_context * cuda_ctx,
        ggml_cgraph * cgraph,
        int node_idx,
        bool use_cuda_graph,
        bool cuda_graph_update_required) {
    return try_rms_mul_mmq_fusion(cuda_ctx, cgraph, node_idx, 
                                   use_cuda_graph, cuda_graph_update_required);
}

static inline bool gfx906_is_mul_handled_by_fusion(
        ggml_backend_cuda_context * cuda_ctx,
        ggml_tensor * node) {
    return is_mul_handled_by_fusion(cuda_ctx, node);
}

static inline bool gfx906_try_prequantized_mul_mat(
        ggml_backend_cuda_context * cuda_ctx,
        ggml_tensor * node) {
    return try_prequantized_mul_mat(cuda_ctx, node);
}
#endif // GFX906_KVQ_MOE_CACHE_ENABLED

#endif // GGML_USE_HIP
