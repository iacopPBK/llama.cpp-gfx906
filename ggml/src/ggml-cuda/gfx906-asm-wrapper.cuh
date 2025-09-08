#pragma once

// GFX906 Assembly Kernel Wrappers
// This file provides C++ wrappers for native GCN assembly kernels

#ifdef GGML_HIP_GFX906_OPTIMIZED

#include <hip/hip_runtime.h>

// Declare external assembly kernels
extern "C" {
    // Wave reduction using native DS_SWIZZLE_B32
    __global__ void wave_reduce_sum_asm(const float* input, float* output);
    
    // XOR shuffle using DS_SWIZZLE_B32
    __global__ void wave_shuffle_xor_asm(const float* input, float* output, int mask);
    
    // Broadcast using DS_BPERMUTE_B32
    __global__ void wave_broadcast_asm(const float* input, float* output, int src_lane);
}

namespace gfx906_asm {

// Template wrapper for assembly wave reduce
template<typename T>
inline void launch_wave_reduce_asm(const T* input, T* output, int n, hipStream_t stream = 0) {
    static_assert(sizeof(T) == 4, "Assembly kernels only support 32-bit types");
    
    // Calculate grid dimensions
    int blocks = (n + 63) / 64;  // One wave per block for simplicity
    
    // Launch assembly kernel
    hipLaunchKernelGGL(wave_reduce_sum_asm, 
                       dim3(blocks), dim3(64), 0, stream,
                       reinterpret_cast<const float*>(input),
                       reinterpret_cast<float*>(output));
}

// Inline assembly version for direct use in device code
// This allows us to use DS_SWIZZLE_B32 directly in CUDA/HIP kernels
__device__ __forceinline__ float ds_swizzle_xor_1(float value) {
    float result;
    // Use inline GCN assembly
    asm volatile(
        "ds_swizzle_b32 %0, %1 offset:0x0041\n\t"
        "s_waitcnt lgkmcnt(0)"
        : "=v"(result)
        : "v"(value)
    );
    return result;
}

__device__ __forceinline__ float ds_swizzle_xor_2(float value) {
    float result;
    asm volatile(
        "ds_swizzle_b32 %0, %1 offset:0x0042\n\t"
        "s_waitcnt lgkmcnt(0)"
        : "=v"(result)
        : "v"(value)
    );
    return result;
}

__device__ __forceinline__ float ds_swizzle_xor_4(float value) {
    float result;
    asm volatile(
        "ds_swizzle_b32 %0, %1 offset:0x0044\n\t"
        "s_waitcnt lgkmcnt(0)"
        : "=v"(result)
        : "v"(value)
    );
    return result;
}

__device__ __forceinline__ float ds_swizzle_xor_8(float value) {
    float result;
    asm volatile(
        "ds_swizzle_b32 %0, %1 offset:0x0048\n\t"
        "s_waitcnt lgkmcnt(0)"
        : "=v"(result)
        : "v"(value)
    );
    return result;
}

__device__ __forceinline__ float ds_swizzle_xor_16(float value) {
    float result;
    asm volatile(
        "ds_swizzle_b32 %0, %1 offset:0x0050\n\t"
        "s_waitcnt lgkmcnt(0)"
        : "=v"(result)
        : "v"(value)
    );
    return result;
}

// Wave reduce using inline assembly DS_SWIZZLE
__device__ __forceinline__ float wave_reduce_sum_asm_inline(float value) {
    // Reduction within 32-thread groups using DS_SWIZZLE
    value += ds_swizzle_xor_1(value);
    value += ds_swizzle_xor_2(value);
    value += ds_swizzle_xor_4(value);
    value += ds_swizzle_xor_8(value);
    value += ds_swizzle_xor_16(value);
    
    // For full 64-thread reduction, we need cross-wave communication
    // Use __shfl_xor for the final step (lanes 0-31 <-> 32-63)
    value += __shfl_xor(value, 32, 64);
    
    return value;
}

// DS_BPERMUTE for arbitrary lane broadcast
__device__ __forceinline__ float ds_bpermute(float value, int src_lane) {
    float result;
    int addr = src_lane * 4;  // Byte address
    
    asm volatile(
        "ds_bpermute_b32 %0, %1, %2\n\t"
        "s_waitcnt lgkmcnt(0)"
        : "=v"(result)
        : "v"(addr), "v"(value)
    );
    return result;
}

// DS_PERMUTE for forward permutation
__device__ __forceinline__ float ds_permute(float value, int dst_lane) {
    float result;
    int addr = dst_lane * 4;  // Byte address
    
    asm volatile(
        "ds_permute_b32 %0, %1, %2\n\t"
        "s_waitcnt lgkmcnt(0)"
        : "=v"(result)
        : "v"(addr), "v"(value)
    );
    return result;
}

// Special swizzle patterns from Vega ISA
__device__ __forceinline__ float ds_swizzle_reverse32(float value) {
    float result;
    asm volatile(
        "ds_swizzle_b32 %0, %1 offset:0x001F\n\t"  // REVERSEX32
        "s_waitcnt lgkmcnt(0)"
        : "=v"(result)
        : "v"(value)
    );
    return result;
}

__device__ __forceinline__ float ds_swizzle_bcast2(float value) {
    float result;
    asm volatile(
        "ds_swizzle_b32 %0, %1 offset:0x0061\n\t"  // BCASTX2
        "s_waitcnt lgkmcnt(0)"
        : "=v"(result)
        : "v"(value)
    );
    return result;
}

__device__ __forceinline__ float ds_swizzle_bcast4(float value) {
    float result;
    asm volatile(
        "ds_swizzle_b32 %0, %1 offset:0x0063\n\t"  // BCASTX4
        "s_waitcnt lgkmcnt(0)"
        : "=v"(result)
        : "v"(value)
    );
    return result;
}

// FFT butterfly patterns
__device__ __forceinline__ float ds_swizzle_fft(float value, int stage) {
    float result;
    // FFT mode: offset >= 0xE000
    // Pattern depends on stage
    int offset = 0xE000 | (stage & 0x1F);
    
    // Note: This needs compile-time constant, so we'd need separate functions
    // for each FFT stage or use template specialization
    asm volatile(
        "ds_swizzle_b32 %0, %1 offset:0xE001\n\t"  // FFT stage 1
        "s_waitcnt lgkmcnt(0)"
        : "=v"(result)
        : "v"(value)
    );
    return result;
}

} // namespace gfx906_asm

#endif // GGML_HIP_GFX906_OPTIMIZED