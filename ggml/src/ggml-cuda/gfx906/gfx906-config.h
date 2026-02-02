#pragma once

// GFX906 (Vega 20 / MI50) kernel configuration

#ifdef GGML_USE_HIP
#define GFX906_MMQ_ITER_K 256
#define GFX906_MMQ_NWARPS 2
#define GFX906_FATTN_Q8_ENABLED 1
#define GFX906_Q8_SUPPORTS_HEAD_DIM(d) \
    ((d) % 32 == 0 && (d) != 40 && (d) != 80 && (d) != 112)

#define GFX906_USE_DPP_REDUCTIONS 1
#define GFX906_FATTN_TILE_SIZE_DEFAULT 128
#define GFX906_Q8_SCALE_HOISTING 1
#define GFX906_KVQ_MOE_CACHE_ENABLED 1
#define GFX906_ROPE_ENABLED 1

// ============================================================================
// Universal Shuffle Primitives
// ============================================================================

// Include DPP functions for AMD
#include "gfx906-warp.cuh"

// Universal shuffle macro - uses DPP on AMD, __shfl_xor_sync on NVIDIA
// Note: This is only defined for HIP builds. For CUDA builds, the caller
// should use __shfl_xor_sync directly.
#define GGML_CUDA_SHFL_XOR(val, offset, width) gfx906_shfl_xor_sync<width>(val, offset)

#endif // GGML_USE_HIP
