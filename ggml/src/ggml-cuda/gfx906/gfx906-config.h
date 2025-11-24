#pragma once

// ============================================================================
// GFX906 (MI50/MI60) Optimization Configuration
// ============================================================================
// This file contains all compile-time configuration for AMD GFX906 optimizations
// Isolates hardware-specific settings from upstream code

#ifdef GGML_USE_HIP

// -----------------------------------------------------------------------------
// Flash Attention Split-K Configuration
// -----------------------------------------------------------------------------
// GFX906 achieves best performance with Split-K disabled
// +3.8% improvement over Split-K enabled (2135 t/s vs 2050 t/s)
#define GFX906_FATTN_SPLIT_K_ENABLED 0

#if GFX906_FATTN_SPLIT_K_ENABLED
    #define GFX906_FATTN_N_SPLIT_MAX 8
#else
    #define GFX906_FATTN_N_SPLIT_MAX 1  // Disable Split-K for optimal performance
#endif

// -----------------------------------------------------------------------------
// MMQ (Matrix Multiply Quantized) Configuration
// -----------------------------------------------------------------------------
#define GFX906_MMQ_ITER_K 256
#define GFX906_MMQ_NWARPS 2

// -----------------------------------------------------------------------------
// Q8 Flash Attention Configuration
// -----------------------------------------------------------------------------
// Enable Q8 quantized flash attention kernel
#define GFX906_FATTN_Q8_ENABLED 1

// Q8 kernel only supports head dimensions that are multiples of 32
// Currently unsupported: 40, 80, 112 (due to shared memory layout constraints)
#define GFX906_Q8_SUPPORTS_HEAD_DIM(d) \
    ((d) % 32 == 0 && (d) != 40 && (d) != 80 && (d) != 112)

// -----------------------------------------------------------------------------
// DPP (Data Parallel Primitives) Configuration
// -----------------------------------------------------------------------------
// Enable AMD DPP-based warp reductions (faster than __shfl on GFX906)
#define GFX906_USE_DPP_REDUCTIONS 1

// -----------------------------------------------------------------------------
// Performance Tuning Parameters
// -----------------------------------------------------------------------------
// These can be tuned for different workloads

// Flash attention tile sizes (leave as default for now)
#define GFX906_FATTN_TILE_SIZE_DEFAULT 128

// Quantization parameters
#define GFX906_Q8_SCALE_HOISTING 1  // Hoist Q_scale computation out of inner loop

#endif // GGML_USE_HIP
