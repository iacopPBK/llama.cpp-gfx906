// gfx906/mmid-helpers.cuh
// Helper functions for mmid.cu GFX906 optimizations

#pragma once

#include "../common.cuh"

#if defined(GGML_USE_HIP)

// On AMD wavefront64 GPUs (like MI50/gfx906), the optimized paths use sub-warp shuffles
// that don't work correctly when n_expert_used >= warp_size/2 (the sub-warp width).
// This function checks if we need to fall back to the generic path.
// Returns true if fallback to generic path is required.
static inline bool gfx906_mmid_needs_generic_fallback(const int n_expert_used) {
    const int id = ggml_cuda_get_device();
    const int warp_size = ggml_cuda_info().devices[id].warp_size;
    return n_expert_used >= warp_size / 2;
}

#else

// No-op version for non-HIP builds
static inline bool gfx906_mmid_needs_generic_fallback(const int n_expert_used) {
    (void)n_expert_used;
    return false;
}

#endif // defined(GGML_USE_HIP)
