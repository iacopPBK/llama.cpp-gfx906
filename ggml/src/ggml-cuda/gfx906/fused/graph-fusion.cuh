#pragma once

// Graph fusion for Q8 KV/MoE caching - detects RMS_NORM -> MUL -> MUL_MAT patterns
// Fusion decisions are cached per-graph to avoid repeated O(n²) scans

#include "../quantize/q8-cache.cuh"
#include "norm-fused-q8.cuh"
#include "mmq-prequantized.cuh"
#include "../../common.cuh"
#include <unordered_map>
#include <unordered_set>

#if defined(GGML_USE_HIP) && GFX906_KVQ_MOE_CACHE_ENABLED

// Cached fusion decision for a node
struct fusion_decision {
    bool should_fuse;
    mmq_q8_1_ds_layout ds_layout;
    const ggml_tensor* mul_node;        // The MUL node following RMS_NORM
    const ggml_tensor* mul_weight_src;  // The weight tensor for MUL
};

// Cache of fusion decisions per graph (keyed by graph pointer + n_nodes for invalidation)
struct fusion_cache {
    const ggml_cgraph* graph_ptr;
    int n_nodes;
    std::unordered_map<int, fusion_decision> decisions;  // node_idx -> decision
    std::unordered_set<const ggml_tensor*> fusible_mul_nodes;
};

// Thread-local fusion cache (one per context would be better, but this is simpler)
static thread_local fusion_cache g_fusion_cache = {nullptr, 0, {}, {}};

// Debug flag - set via environment variable GFX906_FUSION_DEBUG=1
static inline bool fusion_debug_enabled() {
    static int enabled = -1;
    if (enabled == -1) {
        const char* env = getenv("GFX906_FUSION_DEBUG");
        enabled = (env != nullptr && env[0] == '1') ? 1 : 0;
    }
    return enabled == 1;
}

// Analyze graph and cache all fusion decisions (called once per new graph)
static void analyze_graph_for_fusion(ggml_cgraph* cgraph, int cc) {
    const bool debug = fusion_debug_enabled();

    g_fusion_cache.graph_ptr = cgraph;
    g_fusion_cache.n_nodes = cgraph->n_nodes;
    g_fusion_cache.decisions.clear();
    g_fusion_cache.fusible_mul_nodes.clear();

    if (debug) {
        fprintf(stderr, "\n[FUSION] Analyzing graph with %d nodes for fusion opportunities...\n", cgraph->n_nodes);
    }

    // Build consumer map once: tensor -> list of (consumer_node, consumer_idx)
    std::unordered_map<const ggml_tensor*, std::vector<std::pair<ggml_tensor*, int>>> consumer_map;
    for (int j = 0; j < cgraph->n_nodes; j++) {
        ggml_tensor* node = cgraph->nodes[j];
        for (int s = 0; s < GGML_MAX_SRC && node->src[s]; s++) {
            consumer_map[node->src[s]].push_back({node, s});
        }
    }

    int fusion_count = 0;

    // Scan for fusible patterns
    for (int i = 0; i < cgraph->n_nodes - 1; i++) {
        ggml_tensor* node = cgraph->nodes[i];

        if (node->op != GGML_OP_RMS_NORM) {
            continue;
        }

        ggml_tensor* rms_norm = node;
        ggml_tensor* mul = cgraph->nodes[i + 1];

        // Check pattern: RMS_NORM -> MUL
        bool mul_pattern = (mul->op == GGML_OP_MUL) &&
                           (mul->src[0] == rms_norm || mul->src[1] == rms_norm);
        if (!mul_pattern) {
            continue;
        }

        // Get all consumers of the MUL output from pre-built map
        const auto& mul_consumers = consumer_map[mul];

        // Categorize consumers
        std::vector<ggml_tensor*> mmq_consumers;
        bool has_non_mmq_consumer = false;

        for (const auto& [consumer, src_idx] : mul_consumers) {
            if (consumer->op == GGML_OP_MUL_MAT && src_idx == 1) {
                const ggml_tensor* weights = consumer->src[0];
                if (ggml_cuda_should_use_mmq(weights->type, cc, consumer->ne[1], 1)) {
                    mmq_consumers.push_back(consumer);
                } else {
                    has_non_mmq_consumer = true;
                }
            } else {
                has_non_mmq_consumer = true;
            }
        }

        // Check fusion criteria
        if (mmq_consumers.size() < 2 || has_non_mmq_consumer) {
            continue;
        }

        // Verify types
        bool types_ok = (rms_norm->src[0]->type == GGML_TYPE_F32) &&
                        (rms_norm->type == GGML_TYPE_F32) &&
                        (mul->type == GGML_TYPE_F32);
        if (!types_ok) {
            continue;
        }

        // Get Q8_1 layout and verify consistency
        mmq_q8_1_ds_layout ds_layout = mmq_get_q8_1_ds_layout(mmq_consumers[0]->src[0]->type);
        bool layout_consistent = true;
        for (size_t c = 1; c < mmq_consumers.size(); c++) {
            if (mmq_get_q8_1_ds_layout(mmq_consumers[c]->src[0]->type) != ds_layout) {
                layout_consistent = false;
                break;
            }
        }
        if (!layout_consistent) {
            continue;
        }

        // Determine multiply weight source
        const ggml_tensor* mul_weight_src = (mul->src[0] == rms_norm) ? mul->src[1] : mul->src[0];

        // Cache the positive decision
        fusion_decision decision;
        decision.should_fuse = true;
        decision.ds_layout = ds_layout;
        decision.mul_node = mul;
        decision.mul_weight_src = mul_weight_src;

        g_fusion_cache.decisions[i] = decision;
        g_fusion_cache.fusible_mul_nodes.insert(mul);
        fusion_count++;

        if (debug) {
            fprintf(stderr, "[FUSION] Node %d: '%s' -> '%s' FUSIBLE (%zu MMQ consumers)\n",
                    i, rms_norm->name, mul->name, mmq_consumers.size());
        }
    }

    if (debug) {
        fprintf(stderr, "[FUSION] Analysis complete: %d fusion opportunities found\n\n", fusion_count);
    }
}

// Check if cache is valid for this graph
static inline bool is_cache_valid(const ggml_cgraph* cgraph) {
    return g_fusion_cache.graph_ptr == cgraph &&
           g_fusion_cache.n_nodes == cgraph->n_nodes;
}

// Fast path: check cached decision and execute fusion if applicable
static inline bool try_rms_mul_mmq_fusion(
    ggml_backend_cuda_context* cuda_ctx,
    ggml_cgraph* cgraph,
    int node_idx,
    bool use_cuda_graph,
    bool cuda_graph_update_required
) {
    // Rebuild cache if graph changed
    if (!is_cache_valid(cgraph)) {
        const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
        analyze_graph_for_fusion(cgraph, cc);
    }

    // Fast lookup in cache
    auto it = g_fusion_cache.decisions.find(node_idx);
    if (it == g_fusion_cache.decisions.end()) {
        return false;  // Not a fusion candidate
    }

    const fusion_decision& decision = it->second;

    // Skip during CUDA graph capture (pool allocations not allowed)
    if (use_cuda_graph && cuda_graph_update_required) {
        return false;
    }

    // Execute fusion
    ggml_tensor* rms_norm = cgraph->nodes[node_idx];
    const ggml_tensor* input = rms_norm->src[0];
    const int64_t ncols = input->ne[0];
    const int64_t nrows = ggml_nrows(input);
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
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

    // Execute fused kernel
    const float* mul_weights = (const float*)decision.mul_weight_src->data;
    ggml_cuda_op_rms_norm_fused_q8_1(*cuda_ctx, rms_norm, buffer_ptr, decision.ds_layout,
                                      mul_weights, decision.mul_weight_src);

    // Store in map for MUL_MAT consumers
    cuda_ctx->fusion_prequant_map[decision.mul_node] = info;
    cuda_ctx->fusion_handled_mul_nodes.insert(decision.mul_node);

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
