#pragma once

// GFX906 (Vega 20 / MI50) kernel configuration

#ifdef GGML_USE_HIP

#define GFX906_FATTN_SPLIT_K_ENABLED 1

#if GFX906_FATTN_SPLIT_K_ENABLED
    #define GFX906_FATTN_N_SPLIT_MAX 8
#else
    #define GFX906_FATTN_N_SPLIT_MAX 1
#endif

#define GFX906_MMQ_ITER_K 256
#define GFX906_MMQ_NWARPS 2

#define GFX906_FATTN_Q8_ENABLED 1
#define GFX906_Q8_SUPPORTS_HEAD_DIM(d) \
    ((d) % 32 == 0 && (d) != 40 && (d) != 80 && (d) != 112)

#define GFX906_USE_DPP_REDUCTIONS 1
#define GFX906_FATTN_TILE_SIZE_DEFAULT 128
#define GFX906_Q8_SCALE_HOISTING 1

// Q8_1 activation caching for KV projections and MoE routing
#define GFX906_KVQ_MOE_CACHE_ENABLED 1

// Optimized RoPE kernel using __sincosf and precomputed theta
#define GFX906_ROPE_ENABLED 1

#endif // GGML_USE_HIP
