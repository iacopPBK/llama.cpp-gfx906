#pragma once

// Optimized add-id kernels for GFX906
// These kernels provide better performance for specific memory layouts

#if defined(GGML_USE_HIP)

#include <cstdint>
#include <algorithm>
#include <hip/hip_runtime.h>

// ============================================================================
// Vectorized Kernel - Uses float4 for 128-bit memory accesses
// Requirements: Contiguous memory, 16-byte aligned, ne00 divisible by 4
// ============================================================================

static __global__ void gfx906_add_id_kernel_vec4(
        const float * __restrict__ src0,
        const float * __restrict__ src1,
        const int32_t * __restrict__ src2,
        float * __restrict__ dst,
        const int ne0,
        const int ne01,
        const int s0_stride,
        const int s0_stride2,
        const int s1_stride,
        const int s2_stride
    ) {
    const int i1 = blockIdx.x;
    const int i2 = blockIdx.y;

    const int i11 = src2[i1 + i2 * s2_stride];

    const int src0_offset = i1 * s0_stride + i2 * s0_stride2;
    const int src1_offset = i11 * s1_stride;
    const int dst_offset = i1 * ne0 + i2 * ne01 * ne0;

    const float4 * __restrict__ src0_vec = reinterpret_cast<const float4 *>(src0 + src0_offset);
    const float4 * __restrict__ src1_vec = reinterpret_cast<const float4 *>(src1 + src1_offset);
    float4 * __restrict__ dst_vec = reinterpret_cast<float4 *>(dst + dst_offset);

    const int ne0_vec = ne0 >> 2;

    for (int i0 = threadIdx.x; i0 < ne0_vec; i0 += blockDim.x) {
        const float4 a = src0_vec[i0];
        const float4 b = src1_vec[i0];
        dst_vec[i0] = make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
    }
}

// ============================================================================
// Contiguous Kernel - Uses __restrict__ for contiguous memory
// Requirements: Contiguous memory layout
// ============================================================================

static __global__ void gfx906_add_id_kernel_contiguous(
        const float * __restrict__ src0,
        const float * __restrict__ src1,
        const int32_t * __restrict__ src2,
        float * __restrict__ dst,
        const int ne0,
        const int ne01,
        const int s0_stride,
        const int s0_stride2,
        const int s1_stride,
        const int s2_stride
    ) {
    const int i1 = blockIdx.x;
    const int i2 = blockIdx.y;

    const int i11 = src2[i1 + i2 * s2_stride];

    const float * __restrict__ src0_row = src0 + i1 * s0_stride + i2 * s0_stride2;
    const float * __restrict__ src1_row = src1 + i11 * s1_stride;
    float * __restrict__ dst_row = dst + i1 * ne0 + i2 * ne01 * ne0;

    for (int i0 = threadIdx.x; i0 < ne0; i0 += blockDim.x) {
        dst_row[i0] = src0_row[i0] + src1_row[i0];
    }
}

// ============================================================================
// Dispatch Function - Selects optimal kernel based on memory properties
// ============================================================================

static inline void gfx906_add_id_dispatch(
        const float * src0_d, const float * src1_d, const int32_t * src2_d, float * dst_d,
        int64_t ne0, int64_t ne1, int64_t ne00, int64_t ne01, int64_t ne02,
        size_t nb01, size_t nb02, size_t nb11, size_t nb21,
        size_t nb10,  // needed for is_contiguous check
        cudaStream_t stream) {

    // Check memory layout properties
    const bool is_contiguous = (nb01 == ne00 * sizeof(float)) &&
                               (nb11 == nb10);  // src1 row stride equals element stride

    const bool is_aligned = ((uintptr_t)src0_d % 16 == 0) &&
                            ((uintptr_t)src1_d % 16 == 0) &&
                            ((uintptr_t)dst_d  % 16 == 0);

    const bool can_vectorize = is_contiguous && is_aligned && (ne00 % 4 == 0) && (ne00 <= INT_MAX);

    const dim3 blocks(ne01, ne02);

    if (can_vectorize) {
        // Use vectorized kernel for aligned, contiguous data
        const int threads_vec4 = std::min((int)(ne00 / 4), 768);
        gfx906_add_id_kernel_vec4<<<blocks, threads_vec4, 0, stream>>>(
            src0_d, src1_d, src2_d, dst_d,
            (int)ne00,
            (int)ne01,
            (int)(nb01 / sizeof(float)),
            (int)(nb02 / sizeof(float)),
            (int)(nb11 / sizeof(float)),
            (int)(nb21 / sizeof(int32_t))
        );
    } else if (is_contiguous && ne00 <= INT_MAX) {
        // Use contiguous kernel for contiguous data (but not vectorizable)
        const int threads = std::min((int)ne00, 768);
        gfx906_add_id_kernel_contiguous<<<blocks, threads, 0, stream>>>(
            src0_d, src1_d, src2_d, dst_d,
            (int)ne00,
            (int)ne01,
            (int)(nb01 / sizeof(float)),
            (int)(nb02 / sizeof(float)),
            (int)(nb11 / sizeof(float)),
            (int)(nb21 / sizeof(int32_t))
        );
    } else {
        // Fall back to reference kernel - caller should handle this case
        // This shouldn't happen if the caller provides a fallback
    }
}

// ============================================================================
// Check if optimized path can be used
// ============================================================================

static inline bool gfx906_add_id_can_use_optimized(
        int64_t ne00,
        size_t nb01, size_t nb11, size_t nb10) {
    const bool is_contiguous = (nb01 == ne00 * sizeof(float)) &&
                               (nb11 == nb10);
    return is_contiguous && (ne00 <= INT_MAX);
}

#endif // GGML_USE_HIP
