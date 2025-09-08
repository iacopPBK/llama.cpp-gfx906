#pragma once

// Proper DS_SWIZZLE implementation for GFX906
// This version uses the native AMD swizzle instruction with compile-time constants

#ifdef GGML_HIP_GFX906_OPTIMIZED

namespace gfx906 {

// DS_SWIZZLE pattern encoding for butterfly reduction
// Pattern: 0x1F means XOR with lane_id (butterfly pattern)
// The encoding is: (pattern << 10) | xor_mask
template<int OFFSET>
struct swizzle_pattern {
    static constexpr int value = (0x1F << 10) | OFFSET;
};

// Template recursion for wave reduction with compile-time constants
template<int OFFSET>
__device__ __forceinline__ float wave_reduce_sum_impl(float value) {
    if constexpr (OFFSET >= 1) {
        // Use the native swizzle instruction with compile-time constant
        value += __builtin_amdgcn_ds_swizzle(value, swizzle_pattern<OFFSET>::value);
        return wave_reduce_sum_impl<OFFSET/2>(value);
    }
    return value;
}

// Specialized version for specific offsets (fully unrolled)
template<>
__device__ __forceinline__ float wave_reduce_sum_impl<32>(float value) {
    value += __builtin_amdgcn_ds_swizzle(value, (0x1F << 10) | 32);
    value += __builtin_amdgcn_ds_swizzle(value, (0x1F << 10) | 16);
    value += __builtin_amdgcn_ds_swizzle(value, (0x1F << 10) | 8);
    value += __builtin_amdgcn_ds_swizzle(value, (0x1F << 10) | 4);
    value += __builtin_amdgcn_ds_swizzle(value, (0x1F << 10) | 2);
    value += __builtin_amdgcn_ds_swizzle(value, (0x1F << 10) | 1);
    return value;
}

// Main entry point
__device__ __forceinline__ float wave_reduce_sum_swizzle(float value) {
    return wave_reduce_sum_impl<32>(value);
}

// Integer version
__device__ __forceinline__ int wave_reduce_sum_swizzle(int value) {
    value += __builtin_amdgcn_ds_swizzle(value, (0x1F << 10) | 32);
    value += __builtin_amdgcn_ds_swizzle(value, (0x1F << 10) | 16);
    value += __builtin_amdgcn_ds_swizzle(value, (0x1F << 10) | 8);
    value += __builtin_amdgcn_ds_swizzle(value, (0x1F << 10) | 4);
    value += __builtin_amdgcn_ds_swizzle(value, (0x1F << 10) | 2);
    value += __builtin_amdgcn_ds_swizzle(value, (0x1F << 10) | 1);
    return value;
}

// Other swizzle patterns for different operations
namespace swizzle_ops {
    // Reverse pattern (reverse lanes in wave)
    __device__ __forceinline__ float reverse(float value) {
        return __builtin_amdgcn_ds_swizzle(value, (0x1E << 10) | 0x3F);
    }
    
    // Rotate pattern
    template<int ROTATE>
    __device__ __forceinline__ float rotate(float value) {
        constexpr int pattern = (0x1D << 10) | (ROTATE & 0x3F);
        return __builtin_amdgcn_ds_swizzle(value, pattern);
    }
    
    // Broadcast from lane 0
    __device__ __forceinline__ float broadcast_lane0(float value) {
        return __builtin_amdgcn_ds_swizzle(value, (0x00 << 10) | 0);
    }
}

// Benchmark comparison function - removed to avoid duplicate symbols
// (can be added to separate .cu file if needed for testing)

}  // namespace gfx906

#endif  // GGML_HIP_GFX906_OPTIMIZED