// gfx906/rope-dispatch.cuh
// Helper dispatch functions for GFX906 RoPE kernel

#pragma once

#include "../common.cuh"
#include "gfx906-config.h"

#if defined(GGML_USE_HIP) && defined(GFX906_ROPE_ENABLED)
#include "attention/rope.cuh"
#endif

// ============================================================================
// GFX906 RoPE Dispatch Helper
// ============================================================================
// Returns true if GFX906 optimized kernel was dispatched, false otherwise.

#if defined(GGML_USE_HIP) && defined(GFX906_ROPE_ENABLED)

template<bool forward, typename T>
static inline bool gfx906_rope_try_dispatch(
        const T * x, T * dst,
        const int ne0, const int ne1, const int ne2,
        const int s1, const int s2, const int n_dims, const int nr,
        const int32_t * pos, const float freq_scale, const float freq_base,
        const float ext_factor, const float attn_factor,
        const float * corr_dims,  // rope_corr_dims.v[2]
        const float * freq_factors,
        const int * sections,     // mrope_sections.v[4]
        const bool is_imrope, cudaStream_t stream) {

    const gfx906_rope_corr_dims & gfx906_corr = 
        reinterpret_cast<const gfx906_rope_corr_dims &>(*corr_dims);
    const gfx906_mrope_sections & gfx906_sects = 
        reinterpret_cast<const gfx906_mrope_sections &>(*sections);

    gfx906_rope_multi_cuda<forward, T>(
        x, dst, ne0, ne1, ne2, s1, s2, n_dims, nr,
        pos, freq_scale, freq_base, ext_factor, attn_factor,
        gfx906_corr, freq_factors, gfx906_sects, is_imrope, stream);

    return true;
}

#else

// No-op version for non-GFX906 builds
template<bool forward, typename T>
static inline bool gfx906_rope_try_dispatch(
        const T *, T *,
        const int, const int, const int,
        const int, const int, const int, const int,
        const int32_t *, const float, const float,
        const float, const float,
        const float *,
        const float *,
        const int *,
        const bool, cudaStream_t) {
    return false;
}

#endif // defined(GGML_USE_HIP) && defined(GFX906_ROPE_ENABLED)
