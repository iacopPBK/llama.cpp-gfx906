#pragma once

#include "gfx906-config.h"

// ============================================================================
// AMD GFX906 MMQ (Matrix Multiply Quantized) Optimizations
// ============================================================================
// Vectorized quantization loads for Q4_0 and Q4_1
// Replaces 8 scalar 32-bit loads with 2 vectorized 128-bit int4 loads
// Improves memory throughput on GFX906 architecture

#if defined(GGML_USE_HIP)

// ============================================================================
// Q4_0 Vectorized Load
// ============================================================================
// Loads 8 integers as two int4 vectors instead of 8 scalar operations
// Parameters:
//   y_qs: Source quantized data
//   base_addr: Base address offset
//   u: Output array (must be int[8])
static __device__ __forceinline__ void gfx906_load_q4_0_quants_vectorized(
    const int * __restrict__ y_qs,
    const int base_addr,
    const int qi,
    int * __restrict__ u) {

    // Vectorized 128-bit int4 loads
    const int4 vec0 = *((const int4 *) &y_qs[base_addr]);
    const int4 vec1 = *((const int4 *) &y_qs[base_addr + qi]);

    // Unpack into output array
    // Even indices: first vector
    // Odd indices: second vector
    u[0] = vec0.x; u[2] = vec0.y; u[4] = vec0.z; u[6] = vec0.w;
    u[1] = vec1.x; u[3] = vec1.y; u[5] = vec1.z; u[7] = vec1.w;
}

// ============================================================================
// Q4_1 Vectorized Load
// ============================================================================
// Same pattern as Q4_0 but for Q4_1 quantization
static __device__ __forceinline__ void gfx906_load_q4_1_quants_vectorized(
    const int * __restrict__ y_qs,
    const int base_addr,
    const int qi,
    int * __restrict__ u) {

    // Vectorized 128-bit int4 loads
    const int4 vec0 = *((const int4 *) &y_qs[base_addr]);
    const int4 vec1 = *((const int4 *) &y_qs[base_addr + qi]);

    // Unpack into output array
    u[0] = vec0.x; u[2] = vec0.y; u[4] = vec0.z; u[6] = vec0.w;
    u[1] = vec1.x; u[3] = vec1.y; u[5] = vec1.z; u[7] = vec1.w;
}

// ============================================================================
// Generic Vectorized Load (for other quantization types)
// ============================================================================
// Can be used for Q5_0, Q5_1, etc. if they follow the same pattern
template<int VDR>
static __device__ __forceinline__ void gfx906_load_quants_vectorized(
    const int * __restrict__ y_qs,
    const int base_addr,
    const int qi,
    int * __restrict__ u) {

    static_assert(VDR == 4, "Only VDR=4 supported for vectorized loads");

    // Vectorized 128-bit int4 loads
    const int4 vec0 = *((const int4 *) &y_qs[base_addr]);
    const int4 vec1 = *((const int4 *) &y_qs[base_addr + qi]);

    // Unpack into output array
    u[0] = vec0.x; u[2] = vec0.y; u[4] = vec0.z; u[6] = vec0.w;
    u[1] = vec1.x; u[3] = vec1.y; u[5] = vec1.z; u[7] = vec1.w;
}

#endif // GGML_USE_HIP
