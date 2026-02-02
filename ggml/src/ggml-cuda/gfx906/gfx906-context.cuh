#pragma once

// GFX906 Context Extensions
// This file provides access to GFX906-specific features (Q8 cache, etc.)
// without modifying common.cuh.
// 
// Usage:
//   - Include this file in .cu files that need GFX906 features
//   - Use gfx906_ctx_ext(ctx) to get extension for a context
//   - The extension is automatically created on first access

#include "gfx906-config.h"

#if defined(GGML_USE_HIP) && GFX906_KVQ_MOE_CACHE_ENABLED

#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <memory>
#include <mutex>

#include "quantize/q8-cache.cuh"

// Forward declarations
template<typename T> struct ggml_cuda_pool_alloc;
struct ggml_tensor;
struct ggml_backend_cuda_context;

// Extension structure with GFX906-specific data
struct gfx906_context_ext {
    q8_hashmap_cache q8_cache;
    std::unordered_map<const ggml_tensor*, prequantized_q8_info> fusion_prequant_map;
    std::unordered_set<const ggml_tensor*> fusion_handled_mul_nodes;
    std::vector<std::unique_ptr<ggml_cuda_pool_alloc<char>>> fusion_q8_buffers;

    void clear_q8_cache() { q8_cache.clear(); }
    
    void free_all() { 
        q8_cache.free_all();
        fusion_prequant_map.clear();
        fusion_handled_mul_nodes.clear();
        fusion_q8_buffers.clear();
    }
};

// Global registry for context extensions
// Maps context pointer to its extension (created on demand)
class gfx906_ext_registry {
public:
    static gfx906_context_ext& get(ggml_backend_cuda_context* ctx) {
        std::lock_guard<std::mutex> lock(get_mutex());
        auto& registry = get_registry();
        auto& ext = registry[ctx];
        if (!ext) {
            ext = std::make_unique<gfx906_context_ext>();
        }
        return *ext;
    }
    
    static void remove(ggml_backend_cuda_context* ctx) {
        std::lock_guard<std::mutex> lock(get_mutex());
        auto& registry = get_registry();
        auto it = registry.find(ctx);
        if (it != registry.end()) {
            it->second->free_all();
            registry.erase(it);
        }
    }

private:
    // Use static methods to ensure single instances across translation units
    static std::mutex& get_mutex() {
        static std::mutex mutex;
        return mutex;
    }
    
    static std::unordered_map<ggml_backend_cuda_context*, std::unique_ptr<gfx906_context_ext>>& get_registry() {
        static std::unordered_map<ggml_backend_cuda_context*, std::unique_ptr<gfx906_context_ext>> registry;
        return registry;
    }
};

// Helper to get extension for a context
inline gfx906_context_ext& gfx906_ctx_ext(ggml_backend_cuda_context& ctx) {
    return gfx906_ext_registry::get(&ctx);
}

// Macros for backward compatibility with existing code
// These replace the direct member access: ctx.q8_cache -> GFX906_Q8_CACHE(ctx)
#define GFX906_Q8_CACHE(ctx) (gfx906_ctx_ext(ctx).q8_cache)
#define GFX906_PREQUANT_MAP(ctx) (gfx906_ctx_ext(ctx).fusion_prequant_map)
#define GFX906_HANDLED_MUL_NODES(ctx) (gfx906_ctx_ext(ctx).fusion_handled_mul_nodes)
#define GFX906_Q8_BUFFERS(ctx) (gfx906_ctx_ext(ctx).fusion_q8_buffers)
#define GFX906_CLEAR_Q8_CACHE(ctx) (gfx906_ctx_ext(ctx).clear_q8_cache())
#define GFX906_FREE_CONTEXT(ctx) (gfx906_ext_registry::remove(&(ctx)))

#else // !GGML_USE_HIP || !GFX906_KVQ_MOE_CACHE_ENABLED

// No-op macros when GFX906 features are disabled
struct gfx906_context_ext {};  // Empty struct for type safety

inline gfx906_context_ext& gfx906_ctx_ext(ggml_backend_cuda_context&) {
    static gfx906_context_ext dummy;
    return dummy;
}

#define GFX906_Q8_CACHE(ctx) (gfx906_ctx_ext(ctx).q8_cache)
#define GFX906_PREQUANT_MAP(ctx) (gfx906_ctx_ext(ctx).fusion_prequant_map)
#define GFX906_HANDLED_MUL_NODES(ctx) (gfx906_ctx_ext(ctx).fusion_handled_mul_nodes)
#define GFX906_Q8_BUFFERS(ctx) (gfx906_ctx_ext(ctx).fusion_q8_buffers)
#define GFX906_CLEAR_Q8_CACHE(ctx) do {} while(0)
#define GFX906_FREE_CONTEXT(ctx) do {} while(0)

#endif // defined(GGML_USE_HIP) && GFX906_KVQ_MOE_CACHE_ENABLED
