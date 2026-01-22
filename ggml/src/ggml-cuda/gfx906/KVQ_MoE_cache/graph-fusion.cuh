#pragma once

// Graph compute loop integration for Q8 KV/MoE caching.
// Detects RMS_NORM -> MUL -> MUL_MAT patterns and applies fusion.

#include "q8-cache.cuh"
#include "norm-fused-q8.cuh"
#include "mmq-prequantized.cuh"
#include "../../common.cuh"

#if defined(GGML_USE_HIP) && GFX906_KVQ_MOE_CACHE_ENABLED

// Debug flag for fusion diagnostics (host-side only, GFX906 doesn't support device printf)
static bool gfx906_fusion_debug_init = false;
static bool gfx906_fusion_debug = false;

static inline void init_fusion_debug() {
    if (!gfx906_fusion_debug_init) {
        gfx906_fusion_debug = (getenv("GGML_CUDA_Q8_CACHE_DEBUG") != nullptr);
        gfx906_fusion_debug_init = true;
        if (gfx906_fusion_debug) {
            fprintf(stderr, "[GFX906_Q8_FUSION] Fusion system initialized (compile-time enabled)\n");
            fflush(stderr);
        }
    }
}

// Try to detect and handle multi-consumer RMS_NORM + MUL + MUL_MAT fusion.
// Returns true if fusion was applied (caller should skip node), false otherwise.
static inline bool try_rms_mul_mmq_fusion(
    ggml_backend_cuda_context* cuda_ctx,
    ggml_cgraph* cgraph,
    int node_idx,
    bool use_cuda_graph,
    bool cuda_graph_update_required
) {
    init_fusion_debug();

    ggml_tensor* node = cgraph->nodes[node_idx];

    if (node->op != GGML_OP_RMS_NORM || node_idx + 1 >= cgraph->n_nodes) {
        return false;
    }

    if (gfx906_fusion_debug) {
        fprintf(stderr, "[GFX906_Q8_FUSION] Found RMS_NORM at node %d: %s\n", node_idx, node->name);
        fflush(stderr);
    }

    ggml_tensor* rms_norm = node;
    ggml_tensor* mul = cgraph->nodes[node_idx + 1];

    // Check pattern: RMS_NORM -> MUL
    bool mul_pattern = (mul->op == GGML_OP_MUL) &&
                       (mul->src[0] == rms_norm || mul->src[1] == rms_norm);
    if (!mul_pattern) {
        if (gfx906_fusion_debug) {
            fprintf(stderr, "[GFX906_Q8_FUSION]   -> next node is %s (%s), not MUL pattern\n",
                    ggml_op_name(mul->op), mul->name);
            fflush(stderr);
        }
        return false;
    }

    if (gfx906_fusion_debug) {
        fprintf(stderr, "[GFX906_Q8_FUSION]   -> MUL pattern found: %s\n", mul->name);
        fflush(stderr);
    }

    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;

    // Find ALL MUL_MAT consumers of the MUL output
    std::vector<ggml_tensor*> mul_mat_consumers;
    int total_mul_mat_checked = 0;
    int mmq_rejected = 0;
    for (int j = node_idx + 2; j < cgraph->n_nodes; j++) {
        ggml_tensor* other = cgraph->nodes[j];
        if (other->op == GGML_OP_MUL_MAT && other->src[1] == mul) {
            total_mul_mat_checked++;
            const ggml_tensor* weights = other->src[0];
            if (ggml_cuda_should_use_mmq(weights->type, cc, other->ne[1], 1)) {
                mul_mat_consumers.push_back(other);
            } else {
                mmq_rejected++;
            }
        }
    }

    if (gfx906_fusion_debug) {
        fprintf(stderr, "[GFX906_Q8_FUSION]   -> MUL_MAT consumers: found=%zu (checked=%d, mmq_rejected=%d)\n",
                mul_mat_consumers.size(), total_mul_mat_checked, mmq_rejected);
        fflush(stderr);
    }

    // Need at least 2 consumers for multi-consumer fusion to be worthwhile
    if (mul_mat_consumers.size() < 2) {
        if (gfx906_fusion_debug) {
            fprintf(stderr, "[GFX906_Q8_FUSION]   -> REJECTED: need >= 2 consumers\n");
            fflush(stderr);
        }
        return false;
    }

    // Check types are compatible
    bool types_ok = (rms_norm->src[0]->type == GGML_TYPE_F32) &&
                    (rms_norm->type == GGML_TYPE_F32) &&
                    (mul->type == GGML_TYPE_F32);
    if (!types_ok) {
        return false;
    }

    // Skip during CUDA graph capture (pool allocations not allowed)
    if (use_cuda_graph && cuda_graph_update_required) {
        return false;
    }

    // Limit buffers to avoid memory issues
    if (cuda_ctx->fusion_q8_buffers.size() >= 45) {
        return false;
    }

    // Get Q8_1 layout from first consumer's weight type
    mmq_q8_1_ds_layout ds_layout = mmq_get_q8_1_ds_layout(mul_mat_consumers[0]->src[0]->type);

    // Verify all consumers use the same ds_layout
    for (size_t c = 1; c < mul_mat_consumers.size(); c++) {
        if (mmq_get_q8_1_ds_layout(mul_mat_consumers[c]->src[0]->type) != ds_layout) {
            return false;  // Different layouts are incompatible
        }
    }

    // Determine multiply weights
    const float* mul_weights = nullptr;
    const ggml_tensor* mul_src = nullptr;
    if (mul->src[0] == rms_norm) {
        mul_weights = (const float*)mul->src[1]->data;
        mul_src = mul->src[1];
    } else {
        mul_weights = (const float*)mul->src[0]->data;
        mul_src = mul->src[0];
    }

    // Compute buffer size
    const ggml_tensor* input = rms_norm->src[0];
    const int64_t ncols = input->ne[0];
    const int64_t nrows = ggml_nrows(input);
    const size_t q8_buffer_size = ggml_cuda_get_q8_1_buffer_size(ncols, nrows, cc);

    if (gfx906_fusion_debug) {
        fprintf(stderr, "[GFX906_Q8_FUSION] APPLYING FUSION: rms=%s mul=%s consumers=%zu\n",
                rms_norm->name, mul->name, mul_mat_consumers.size());
        fprintf(stderr, "[GFX906_Q8_FUSION]   input dims: [%ld,%ld,%ld,%ld] nrows=%ld buffer_size=%zu\n",
                (long)input->ne[0], (long)input->ne[1], (long)input->ne[2], (long)input->ne[3],
                (long)nrows, q8_buffer_size);
        fflush(stderr);
    }

    // Allocate Q8_1 buffer
    auto pool_alloc = std::make_unique<ggml_cuda_pool_alloc<char>>();
    pool_alloc->alloc(cuda_ctx->pool(), q8_buffer_size);
    char* buffer_ptr = pool_alloc->get();
    cuda_ctx->fusion_q8_buffers.push_back(std::move(pool_alloc));

    // Store dimensions
    prequantized_q8_info info;
    info.buffer_ptr = buffer_ptr;
    info.ne10 = input->ne[0];
    info.ne11 = input->ne[1];
    info.ne12 = input->ne[2];
    info.ne13 = input->ne[3];

    // Execute fused RMS_NORM + MUL + Q8_1 kernel
    ggml_cuda_op_rms_norm_fused_q8_1(*cuda_ctx, rms_norm, buffer_ptr, ds_layout, mul_weights, mul_src);

    // Store in map for MUL_MAT consumers to use
    cuda_ctx->fusion_prequant_map[mul] = info;
    cuda_ctx->fusion_handled_mul_nodes.insert(mul);

    if (gfx906_fusion_debug) {
        fprintf(stderr, "[GFX906_Q8_FUSION] Fusion applied, buffer=%p map_size=%zu\n",
                (void*)buffer_ptr, cuda_ctx->fusion_prequant_map.size());
        fflush(stderr);
    }

    return true;
}

// Check if a MUL node was handled by fusion (should be skipped)
static inline bool is_mul_handled_by_fusion(ggml_backend_cuda_context* cuda_ctx, ggml_tensor* node) {
    if (node->op != GGML_OP_MUL) {
        return false;
    }
    return cuda_ctx->fusion_handled_mul_nodes.count(node) > 0;
}

// Try to use prequantized data for MUL_MAT. Returns true if handled.
static inline bool try_prequantized_mul_mat(ggml_backend_cuda_context* cuda_ctx, ggml_tensor* node) {
    if (node->op != GGML_OP_MUL_MAT || node->src[1] == nullptr) {
        return false;
    }

    auto it = cuda_ctx->fusion_prequant_map.find(node->src[1]);
    if (it == cuda_ctx->fusion_prequant_map.end()) {
        return false;
    }

    const prequantized_q8_info& info = it->second;

    // Verify dimensions match
    if (info.ne10 != node->src[1]->ne[0] || info.ne11 != node->src[1]->ne[1] ||
        info.ne12 != node->src[1]->ne[2] || info.ne13 != node->src[1]->ne[3]) {
        if (gfx906_fusion_debug) {
            fprintf(stderr, "[GFX906_Q8_FUSION] PREQUANT dimension mismatch for %s, using regular path\n",
                    node->name);
            fflush(stderr);
        }
        return false;  // Dimension mismatch, fall through to regular path
    }

    if (gfx906_fusion_debug) {
        fprintf(stderr, "[GFX906_Q8_FUSION] Using PREQUANTIZED path for MUL_MAT: %s\n", node->name);
        fflush(stderr);
    }

    // Use prequantized path
    ggml_cuda_mul_mat_q_prequantized(*cuda_ctx, node->src[0], info.buffer_ptr,
                                      node, info.ne10, info.ne11, info.ne12, info.ne13);
    return true;
}

// Clear fusion state at start of graph compute
// NOTE: This does NOT clear q8_cache - that's a separate cache used by mmq.cu
// and needs to persist through the graph evaluation for cross-op reuse
static inline void clear_fusion_state(ggml_backend_cuda_context* cuda_ctx) {
    init_fusion_debug();
    if (gfx906_fusion_debug && !cuda_ctx->fusion_q8_buffers.empty()) {
        fprintf(stderr, "[GFX906_Q8_FUSION] Clearing fusion state: buffers=%zu map=%zu handled=%zu\n",
                cuda_ctx->fusion_q8_buffers.size(),
                cuda_ctx->fusion_prequant_map.size(),
                cuda_ctx->fusion_handled_mul_nodes.size());
        fflush(stderr);
    }
    cuda_ctx->fusion_prequant_map.clear();
    cuda_ctx->fusion_handled_mul_nodes.clear();
    cuda_ctx->fusion_q8_buffers.clear();
    // NOTE: q8_cache.clear() is called separately at graph_compute END,
    // NOT here - the hashmap cache needs to persist during graph evaluation
}

#endif // GGML_USE_HIP && GFX906_KVQ_MOE_CACHE_ENABLED
