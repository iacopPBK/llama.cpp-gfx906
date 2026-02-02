#pragma once

// GFX906 FlashAttention Helper Functions
// Contains AMD-specific optimizations for FlashAttention that can be
// called from fattn-common.cuh without inlining large code blocks.

#include "gfx906-config.h"

#if defined(GGML_USE_HIP)

// ============================================================================
// Split-K Tuning for AMD GPUs
// ============================================================================

// Tune Split-K parallelism for FlashAttention on AMD GPUs
// 
// Strategy:
//   - Prompt Processing (PP, num_query_tokens > 1): Disable Split-K (return 1)
//     to avoid combine kernel overhead and improve cache efficiency
//   - Token Generation (TG, num_query_tokens == 1): Keep auto-tuned value
//     for better memory latency hiding
//
// Parameters:
//   - parallel_blocks: Auto-tuned value from upstream heuristic
//   - cc: Compute capability
//   - num_query_tokens: Q->ne[1], number of query tokens (1 for TG, >1 for PP)
//
// Returns:
//   - Adjusted parallel_blocks value

static inline int gfx906_fattn_tune_split_k(int parallel_blocks, int cc, int64_t num_query_tokens) {
    // Only adjust for AMD GPUs
    if (GGML_CUDA_CC_IS_NVIDIA(cc)) {
        return parallel_blocks;
    }
    
    // PP: Disable Split-K to avoid combine overhead
    // PP has large tiles with significant partial result data
    // Single block per tile keeps KV cache in L2
    if (num_query_tokens > 1) {
        return 1;
    }
    
    // TG: Use auto-tuned value for better SM utilization
    // TG is memory-bound, benefits from parallel memory access
    return parallel_blocks;
}

#endif // GGML_USE_HIP
