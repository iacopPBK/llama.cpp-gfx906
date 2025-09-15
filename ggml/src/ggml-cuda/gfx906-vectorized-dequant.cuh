#pragma once

#include "common.cuh"

#ifdef GGML_USE_HIP
#include <hip/hip_vector_types.h>
#endif

// Custom half4 type for HIP (since HIP doesn't provide native half4)
struct half4 {
    half x, y, z, w;
    
    __device__ __forceinline__ half4() {}
    __device__ __forceinline__ half4(half x_, half y_, half z_, half w_) : x(x_), y(y_), z(z_), w(w_) {}
};

// GFX906 Vectorized Dequantization Functions
// Optimized for 4× throughput improvement over scalar dequantization

// Vectorized Q8_0 dequantization - processes 4 values simultaneously
template <int D>
__device__ __forceinline__ half4 dequantize_4v_gfx906_q8_0(const char * __restrict__ V_base, int tid_base) {
    // Process 4 consecutive head dimensions: tid_base, tid_base+1, tid_base+2, tid_base+3
    const block_q8_0 * V_q8_0 = (const block_q8_0 *) V_base;
    
    half4 result;
    
    // Process each of the 4 values
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int tid = tid_base + i;
        if (tid < D) {
            const int block_idx = tid / QK8_0;        // Which Q8_0 block (0-3 for D=128)
            const int local_idx = tid % QK8_0;        // Index within block (0-31)
            
            if (block_idx < D/QK8_0) {
                const block_q8_0& block = V_q8_0[block_idx];
                const float d = __half2float(block.d);
                const float dequant_val = d * (float)block.qs[local_idx];
                
                // Store in appropriate half4 component
                switch (i) {
                    case 0: result.x = __float2half(dequant_val); break;
                    case 1: result.y = __float2half(dequant_val); break;
                    case 2: result.z = __float2half(dequant_val); break;
                    case 3: result.w = __float2half(dequant_val); break;
                }
            } else {
                // Out of bounds - set to zero
                switch (i) {
                    case 0: result.x = __float2half(0.0f); break;
                    case 1: result.y = __float2half(0.0f); break;
                    case 2: result.z = __float2half(0.0f); break;
                    case 3: result.w = __float2half(0.0f); break;
                }
            }
        } else {
            // Thread ID out of bounds
            switch (i) {
                case 0: result.x = __float2half(0.0f); break;
                case 1: result.y = __float2half(0.0f); break;
                case 2: result.z = __float2half(0.0f); break;
                case 3: result.w = __float2half(0.0f); break;
            }
        }
    }
    
    return result;
}

// CORRECT: Vectorized sequence processing for same head dimension
template <int D>
__device__ __forceinline__ half4 dequantize_4v_sequence_q8_0(
    const char * __restrict__ V_base, 
    int k0,           // Starting sequence position  
    int head_dim,     // Head dimension (same for all 4 values)
    int64_t nb21      // Stride between sequences
) {
    half4 result;
    
    // Process 4 consecutive sequences: k0, k0+1, k0+2, k0+3
    #pragma unroll
    for (int seq_offset = 0; seq_offset < 4; ++seq_offset) {
        const int seq_pos = k0 + seq_offset;
        
        if (seq_pos < D) {
            // Access V[seq_pos, head_dim]
            const char* V_seq = V_base + seq_pos * nb21;
            const block_q8_0 * V_q8_0 = (const block_q8_0 *) V_seq;
            
            const int block_idx = head_dim / QK8_0;
            const int local_idx = head_dim % QK8_0;
            
            if (block_idx < D/QK8_0) {
                const block_q8_0& block = V_q8_0[block_idx];
                const float d = __half2float(block.d);
                const float dequant_val = d * (float)block.qs[local_idx];
                
                switch (seq_offset) {
                    case 0: result.x = __float2half(dequant_val); break;
                    case 1: result.y = __float2half(dequant_val); break;
                    case 2: result.z = __float2half(dequant_val); break;
                    case 3: result.w = __float2half(dequant_val); break;
                }
            } else {
                switch (seq_offset) {
                    case 0: result.x = __float2half(0.0f); break;
                    case 1: result.y = __float2half(0.0f); break;
                    case 2: result.z = __float2half(0.0f); break;
                    case 3: result.w = __float2half(0.0f); break;
                }
            }
        } else {
            switch (seq_offset) {
                case 0: result.x = __float2half(0.0f); break;
                case 1: result.y = __float2half(0.0f); break;
                case 2: result.z = __float2half(0.0f); break;
                case 3: result.w = __float2half(0.0f); break;
            }
        }
    }
    
    return result;
}

// Helper function to extract individual half values from half4
__device__ __forceinline__ half get_half4_component(const half4& vec, int index) {
    switch (index) {
        case 0: return vec.x;
        case 1: return vec.y;
        case 2: return vec.z;
        case 3: return vec.w;
        default: return __float2half(0.0f);
    }
}

// Vectorized dequantization for 2 values (matching current V_k.x, V_k.y pattern)
template <int D>
__device__ __forceinline__ half2 dequantize_2v_gfx906_q8_0(const char * __restrict__ V_base, int tid_base) {
    const block_q8_0 * V_q8_0 = (const block_q8_0 *) V_base;
    
    half2 result;
    
    // Process 2 consecutive values
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int tid = tid_base + i;
        
        if (tid < D) {
            const int block_idx = tid / QK8_0;
            const int local_idx = tid % QK8_0;
            
            if (block_idx < D/QK8_0) {
                const block_q8_0& block = V_q8_0[block_idx];
                const float d = __half2float(block.d);
                const float dequant_val = d * (float)block.qs[local_idx];
                
                if (i == 0) {
                    result.x = __float2half(dequant_val);
                } else {
                    result.y = __float2half(dequant_val);
                }
            } else {
                if (i == 0) {
                    result.x = __float2half(0.0f);
                } else {
                    result.y = __float2half(0.0f);
                }
            }
        } else {
            if (i == 0) {
                result.x = __float2half(0.0f);
            } else {
                result.y = __float2half(0.0f);
            }
        }
    }
    
    return result;
}

// Advanced: Block-level vectorized dequantization with shared scale loading
template <int D>
__device__ __forceinline__ void dequantize_block_vectorized_q8_0(
    const char * __restrict__ V_base, 
    int sequence_offset,
    half * __restrict__ output_buffer,
    int buffer_stride
) {
    const block_q8_0 * V_q8_0 = (const block_q8_0 *) (V_base + sequence_offset * sizeof(block_q8_0) * (D / QK8_0));
    
    const int tid = threadIdx.x + threadIdx.y * blockDim.x;
    
    // Each thread processes multiple elements to improve memory coalescing
    constexpr int ELEMENTS_PER_THREAD = 4;
    
    for (int elem_group = 0; elem_group < D; elem_group += blockDim.x * ELEMENTS_PER_THREAD) {
        const int base_idx = elem_group + tid * ELEMENTS_PER_THREAD;
        
        if (base_idx < D) {
            // Vectorized processing of 4 elements per thread
            #pragma unroll
            for (int i = 0; i < ELEMENTS_PER_THREAD && (base_idx + i) < D; ++i) {
                const int element_idx = base_idx + i;
                const int block_idx = element_idx / QK8_0;
                const int local_idx = element_idx % QK8_0;
                
                const block_q8_0& block = V_q8_0[block_idx];
                const float d = __half2float(block.d);
                const float dequant_val = d * (float)block.qs[local_idx];
                
                output_buffer[element_idx * buffer_stride] = __float2half(dequant_val);
            }
        }
    }
}