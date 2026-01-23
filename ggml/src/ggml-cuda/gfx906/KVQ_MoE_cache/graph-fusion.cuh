#pragma once

// Graph fusion for Q8 KV/MoE caching - detects RMS_NORM -> MUL -> MUL_MAT patterns

#include "q8-cache.cuh"
#include "norm-fused-q8.cuh"
#include "mmq-prequantized.cuh"
#include "../../common.cuh"

#if defined(GGML_USE_HIP) && GFX906_KVQ_MOE_CACHE_ENABLED

// Detect and handle multi-consumer RMS_NORM + MUL + MUL_MAT fusion
// Returns true if fusion was applied (caller should skip node)
static inline bool try_rms_mul_mmq_fusion(
    ggml_backend_cuda_context* cuda_ctx,
    ggml_cgraph* cgraph,
    int node_idx,
    bool use_cuda_graph,
    bool cuda_graph_update_required
) {
    ggml_tensor* node = cgraph->nodes[node_idx];

    if (node->op != GGML_OP_RMS_NORM || node_idx + 1 >= cgraph->n_nodes) {
        return false;
    }

    ggml_tensor* rms_norm = node;
    ggml_tensor* mul = cgraph->nodes[node_idx + 1];

    // Check pattern: RMS_NORM -> MUL
    bool mul_pattern = (mul->op == GGML_OP_MUL) &&
                       (mul->src[0] == rms_norm || mul->src[1] == rms_norm);
    if (!mul_pattern) {
        return false;
    }

    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;

    // Find all MUL_MAT consumers of the MUL output that should use MMQ
    std::vector<ggml_tensor*> mul_mat_consumers;
    for (int j = node_idx + 2; j < cgraph->n_nodes; j++) {
        ggml_tensor* other = cgraph->nodes[j];
        if (other->op == GGML_OP_MUL_MAT && other->src[1] == mul) {
            const ggml_tensor* weights = other->src[0];
            if (ggml_cuda_should_use_mmq(weights->type, cc, other->ne[1], 1)) {
                mul_mat_consumers.push_back(other);
            }
        }
    }

    // Need at least 2 consumers for multi-consumer fusion
    if (mul_mat_consumers.size() < 2) {
        return false;
    }

    // Verify types
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

    // Get Q8_1 layout from first consumer's weight type
    mmq_q8_1_ds_layout ds_layout = mmq_get_q8_1_ds_layout(mul_mat_consumers[0]->src[0]->type);

    // Verify all consumers use the same ds_layout
    for (size_t c = 1; c < mul_mat_consumers.size(); c++) {
        if (mmq_get_q8_1_ds_layout(mul_mat_consumers[c]->src[0]->type) != ds_layout) {
            return false;
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

    // Compute buffer size and allocate
    const ggml_tensor* input = rms_norm->src[0];
    const int64_t ncols = input->ne[0];
    const int64_t nrows = ggml_nrows(input);
    const size_t q8_buffer_size = ggml_cuda_get_q8_1_buffer_size(ncols, nrows, cc);

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

    // Store in map for MUL_MAT consumers
    cuda_ctx->fusion_prequant_map[mul] = info;
    cuda_ctx->fusion_handled_mul_nodes.insert(mul);

    return true;
}

// Check if a MUL node was handled by fusion (should be skipped)
static inline bool is_mul_handled_by_fusion(ggml_backend_cuda_context* cuda_ctx, ggml_tensor* node) {
    if (node->op != GGML_OP_MUL) {
        return false;
    }
    return cuda_ctx->fusion_handled_mul_nodes.count(node) > 0;
}

// Use prequantized data for MUL_MAT if available
static inline bool try_prequantized_mul_mat(ggml_backend_cuda_context* cuda_ctx, ggml_tensor* node) {
    if (node->op != GGML_OP_MUL_MAT || node->src[1] == nullptr) {
        return false;
    }

    // Verify the weight type supports MMQ before using prequantized path
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    if (!ggml_cuda_should_use_mmq(node->src[0]->type, cc, node->ne[1], 1)) {
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
        return false;
    }

    ggml_cuda_mul_mat_q_prequantized(*cuda_ctx, node->src[0], info.buffer_ptr,
                                      node, info.ne10, info.ne11, info.ne12, info.ne13);
    return true;
}

// Clear fusion state at start of graph compute (NOT q8_cache - that persists for cross-op reuse)
static inline void clear_fusion_state(ggml_backend_cuda_context* cuda_ctx) {
    cuda_ctx->fusion_prequant_map.clear();
    cuda_ctx->fusion_handled_mul_nodes.clear();
    cuda_ctx->fusion_q8_buffers.clear();
}

#endif // GGML_USE_HIP && GFX906_KVQ_MOE_CACHE_ENABLED
