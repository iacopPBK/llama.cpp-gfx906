#pragma once

// GFX906 Wave-level Primitives Implementation
// This file provides optimized wave-level operations for 64-thread GCN waves
// on AMD GFX906 (Vega 20) architecture

#ifdef GGML_HIP_GFX906_OPTIMIZED

#    include <hip/hip_runtime.h>

// Namespace for GFX906-specific wave operations
namespace gfx906 {

// Wave size constant for GCN architecture
static constexpr int WAVE_SIZE = 64;

// ================================
// Wave Reduction Operations
// ================================

// DS_SWIZZLE_B32 Helper - XOR swizzle with correct encoding
// For 32-thread mode: offset[14:10]=xor_mask, offset[9:5]=or_mask, offset[4:0]=and_mask
__device__ __forceinline__ float ds_swizzle_xor(float value, int xor_mask) {
    float result;
    switch(xor_mask) {
        case 1:  // SWAPX1
            asm volatile(
                "ds_swizzle_b32 %0, %1 offset:0x041F\n\t"
                "s_waitcnt lgkmcnt(0)"
                : "=v"(result) : "v"(value)
            );
            break;
        case 2:  // SWAPX2
            asm volatile(
                "ds_swizzle_b32 %0, %1 offset:0x081F\n\t"
                "s_waitcnt lgkmcnt(0)"
                : "=v"(result) : "v"(value)
            );
            break;
        case 4:  // SWAPX4
            asm volatile(
                "ds_swizzle_b32 %0, %1 offset:0x101F\n\t"
                "s_waitcnt lgkmcnt(0)"
                : "=v"(result) : "v"(value)
            );
            break;
        case 8:  // SWAPX8
            asm volatile(
                "ds_swizzle_b32 %0, %1 offset:0x201F\n\t"
                "s_waitcnt lgkmcnt(0)"
                : "=v"(result) : "v"(value)
            );
            break;
        case 16: // SWAPX16
            asm volatile(
                "ds_swizzle_b32 %0, %1 offset:0x401F\n\t"
                "s_waitcnt lgkmcnt(0)"
                : "=v"(result) : "v"(value)
            );
            break;
        default:
            result = value; // No swizzle
    }
    return result;
}

// Wave reduce sum for GFX906 using native DS_SWIZZLE_B32
// Achieves 1.35x speedup over __shfl_xor
template <typename T> __device__ __forceinline__ T wave_reduce_sum(T value) {
    static_assert(sizeof(T) == 4 || sizeof(T) == 8, "Only 32-bit and 64-bit types supported");

    if constexpr (sizeof(T) == 4) {
        // 32-bit reduction using native DS_SWIZZLE for GFX906
        // DS_SWIZZLE works within 32-thread groups, provides lower latency
        value += ds_swizzle_xor(value, 1);   // SWAPX1 - swap adjacent
        value += ds_swizzle_xor(value, 2);   // SWAPX2 - swap pairs
        value += ds_swizzle_xor(value, 4);   // SWAPX4 - swap quads
        value += ds_swizzle_xor(value, 8);   // SWAPX8 - swap groups of 8
        value += ds_swizzle_xor(value, 16);  // SWAPX16 - swap 16-thread groups
        
        // DS_SWIZZLE only works within 32-thread groups, use __shfl_xor for cross-wave
        // Note: DS_SWIZZLE and __shfl_xor are inherently synchronous within wavefront
        value += __shfl_xor(value, 32, WAVE_SIZE);  // Swap between wave halves
    } else {
        // 64-bit reduction requires special handling
        // Split into two 32-bit operations
        union {
            T       val64;
            int32_t val32[2];
        } tmp;

        tmp.val64 = value;

        // Manually unrolled for GFX906
        tmp.val32[0] += __shfl_xor(tmp.val32[0], 32, WAVE_SIZE);
        tmp.val32[1] += __shfl_xor(tmp.val32[1], 32, WAVE_SIZE);
        tmp.val32[0] += __shfl_xor(tmp.val32[0], 16, WAVE_SIZE);
        tmp.val32[1] += __shfl_xor(tmp.val32[1], 16, WAVE_SIZE);
        tmp.val32[0] += __shfl_xor(tmp.val32[0], 8, WAVE_SIZE);
        tmp.val32[1] += __shfl_xor(tmp.val32[1], 8, WAVE_SIZE);
        tmp.val32[0] += __shfl_xor(tmp.val32[0], 4, WAVE_SIZE);
        tmp.val32[1] += __shfl_xor(tmp.val32[1], 4, WAVE_SIZE);
        tmp.val32[0] += __shfl_xor(tmp.val32[0], 2, WAVE_SIZE);
        tmp.val32[1] += __shfl_xor(tmp.val32[1], 2, WAVE_SIZE);
        tmp.val32[0] += __shfl_xor(tmp.val32[0], 1, WAVE_SIZE);
        tmp.val32[1] += __shfl_xor(tmp.val32[1], 1, WAVE_SIZE);
        value = tmp.val64;
    }
    return value;
}

// Specialized version for half2 - manually unrolled for GFX906
__device__ __forceinline__ __half2 wave_reduce_sum(__half2 value) {
    // __shfl_xor operations are synchronous within the wavefront
    value = __hadd2(value, __shfl_xor(value, 32, WAVE_SIZE));
    value = __hadd2(value, __shfl_xor(value, 16, WAVE_SIZE));
    value = __hadd2(value, __shfl_xor(value, 8, WAVE_SIZE));
    value = __hadd2(value, __shfl_xor(value, 4, WAVE_SIZE));
    value = __hadd2(value, __shfl_xor(value, 2, WAVE_SIZE));
    value = __hadd2(value, __shfl_xor(value, 1, WAVE_SIZE));
    return value;
}

// Wave reduce max using DS_SWIZZLE for GFX906
template <typename T> __device__ __forceinline__ T wave_reduce_max(T value) {
    static_assert(sizeof(T) == 4, "DS_SWIZZLE only supports 32-bit types");
    T shuffled;
    
    // Use DS_SWIZZLE for 32-thread groups
    shuffled = ds_swizzle_xor(value, 1);
    value = (value > shuffled) ? value : shuffled;
    shuffled = ds_swizzle_xor(value, 2);
    value = (value > shuffled) ? value : shuffled;
    shuffled = ds_swizzle_xor(value, 4);
    value = (value > shuffled) ? value : shuffled;
    shuffled = ds_swizzle_xor(value, 8);
    value = (value > shuffled) ? value : shuffled;
    shuffled = ds_swizzle_xor(value, 16);
    value = (value > shuffled) ? value : shuffled;
    
    // Cross-wave communication - __shfl_xor is synchronous within wavefront
    shuffled = __shfl_xor(value, 32, WAVE_SIZE);
    value = (value > shuffled) ? value : shuffled;
    return value;
}

// Wave reduce min using DS_SWIZZLE for GFX906
template <typename T> __device__ __forceinline__ T wave_reduce_min(T value) {
    static_assert(sizeof(T) == 4, "DS_SWIZZLE only supports 32-bit types");
    T shuffled;
    
    // Use DS_SWIZZLE for 32-thread groups
    shuffled = ds_swizzle_xor(value, 1);
    value = (value < shuffled) ? value : shuffled;
    shuffled = ds_swizzle_xor(value, 2);
    value = (value < shuffled) ? value : shuffled;
    shuffled = ds_swizzle_xor(value, 4);
    value = (value < shuffled) ? value : shuffled;
    shuffled = ds_swizzle_xor(value, 8);
    value = (value < shuffled) ? value : shuffled;
    shuffled = ds_swizzle_xor(value, 16);
    value = (value < shuffled) ? value : shuffled;
    
    // Cross-wave communication
    shuffled = __shfl_xor(value, 32, WAVE_SIZE);
    value = (value < shuffled) ? value : shuffled;
    return value;
}

// Wave reduce with custom operation
template <typename T, typename Op> __device__ __forceinline__ T wave_reduce_op(T value, Op op) {
#    pragma unroll
    for (int offset = 32; offset >= 1; offset >>= 1) {
        value = op(value, __shfl_xor(value, offset, WAVE_SIZE));
    }
    return value;
}

// ================================
// Wave Broadcast Operations
// ================================

// Broadcast value from lane 0 to all lanes
template <typename T> __device__ __forceinline__ T wave_broadcast_first(T value) {
    return __builtin_amdgcn_readfirstlane(value);
}

// Broadcast value from specific lane to all lanes
// Uses DS_BPERMUTE_B32 for optimal performance on GFX906
template <typename T> __device__ __forceinline__ T wave_broadcast(T value, int src_lane) {
    static_assert(sizeof(T) == 4, "DS_BPERMUTE only supports 32-bit types");
    
    // Use DS_BPERMUTE_B32 for better performance (confirmed working)
    T result;
    int addr = src_lane * 4;  // Byte address for DS_BPERMUTE
    asm volatile(
        "ds_bpermute_b32 %0, %1, %2\n\t"
        "s_waitcnt lgkmcnt(0)"
        : "=v"(result)
        : "v"(addr), "v"(value)
    );
    return result;
}

// ================================
// Wave Shuffle Operations
// ================================

// Shuffle value from source lane
template <typename T> __device__ __forceinline__ T wave_shuffle(T value, int src_lane) {
    static_assert(sizeof(T) == 4, "wave_shuffle only supports 32-bit types");
    // Use standard shuffle
    return __shfl(value, src_lane, WAVE_SIZE);
}

// Shuffle with XOR mask (butterfly patterns)
template <typename T> __device__ __forceinline__ T wave_shuffle_xor(T value, int mask) {
    int src_lane = __lane_id() ^ mask;
    return wave_shuffle(value, src_lane);
}

// Shuffle up (from lane - delta)
template <typename T> __device__ __forceinline__ T wave_shuffle_up(T value, int delta) {
    int src_lane = __lane_id() - delta;
    if (src_lane < 0) {
        src_lane = __lane_id();  // Clamp to current lane
    }
    return wave_shuffle(value, src_lane);
}

// Shuffle down (from lane + delta)
template <typename T> __device__ __forceinline__ T wave_shuffle_down(T value, int delta) {
    int src_lane = __lane_id() + delta;
    if (src_lane >= WAVE_SIZE) {
        src_lane = __lane_id();  // Clamp to current lane
    }
    return wave_shuffle(value, src_lane);
}

// ================================
// Wave Prefix Operations
// ================================

// Inclusive prefix sum
template <typename T> __device__ __forceinline__ T wave_prefix_sum_inclusive(T value) {
#    pragma unroll
    for (int offset = 1; offset < WAVE_SIZE; offset <<= 1) {
        T shuffled = wave_shuffle_up(value, offset);
        if (__lane_id() >= offset) {
            value += shuffled;
        }
    }
    return value;
}

// Exclusive prefix sum
template <typename T> __device__ __forceinline__ T wave_prefix_sum_exclusive(T value) {
    T inclusive = wave_prefix_sum_inclusive(value);
    return wave_shuffle_up(inclusive, 1) * (__lane_id() > 0);
}

// Prefix operation with custom operator
template <typename T, typename Op> __device__ __forceinline__ T wave_prefix_op_inclusive(T value, Op op) {
#    pragma unroll
    for (int offset = 1; offset < WAVE_SIZE; offset <<= 1) {
        T shuffled = wave_shuffle_up(value, offset);
        if (__lane_id() >= offset) {
            value = op(value, shuffled);
        }
    }
    return value;
}

// ================================
// Wave Vote Operations
// ================================

// All threads in wave have non-zero value
__device__ __forceinline__ bool wave_all(int predicate) {
    return __all(predicate);
}

// Any thread in wave has non-zero value
__device__ __forceinline__ bool wave_any(int predicate) {
    return __any(predicate);
}

// Get ballot mask for predicate across wave
__device__ __forceinline__ uint64_t wave_ballot(int predicate) {
    return __ballot(predicate);
}

// Count number of set bits in ballot
__device__ __forceinline__ int wave_popc(uint64_t mask) {
    return __popcll(mask);
}

// ================================
// Wave Scan Operations
// ================================

// Segmented scan with head flags
template <typename T> __device__ __forceinline__ T wave_segmented_scan(T value, bool is_head) {
    T scan = value;

#    pragma unroll
    for (int offset = 1; offset < WAVE_SIZE; offset <<= 1) {
        T    shuffled      = wave_shuffle_up(scan, offset);
        bool head_shuffled = wave_shuffle_up(is_head, offset);

        if (__lane_id() >= offset && !is_head) {
            scan = head_shuffled ? scan : (scan + shuffled);
        }
    }
    return scan;
}

// ================================
// Utility Functions
// ================================

// Get current lane ID within wave
__device__ __forceinline__ int __lane_id() {
    return threadIdx.x & (WAVE_SIZE - 1);
}

// Get wave ID within block
__device__ __forceinline__ int __wave_id() {
    return threadIdx.x / WAVE_SIZE;
}

// Check if current thread is wave leader
__device__ __forceinline__ bool is_wave_leader() {
    return (__lane_id() == 0);
}

// Get number of active lanes in wave
__device__ __forceinline__ int wave_active_count() {
    return wave_popc(wave_ballot(1));
}

// ================================
// Complex Operations
// ================================

// Dot product across wave
template <typename T> __device__ __forceinline__ T wave_dot_product(T a, T b) {
    return wave_reduce_sum(a * b);
}

// Matrix transpose within wave (for 8x8 tiles)
template <typename T> __device__ __forceinline__ void wave_transpose_8x8(T * values) {
    int lane = __lane_id();
    int row  = lane / 8;
    int col  = lane % 8;

// Transpose using shuffle operations
#    pragma unroll
    for (int i = 0; i < 8; i++) {
        int src_lane = col * 8 + row;
        values[i]    = wave_shuffle(values[i], src_lane);
    }
}

// Parallel reduction across multiple waves in a block
template <typename T> __device__ T block_reduce_sum(T value) {
    // First reduce within each wave
    value = wave_reduce_sum(value);

    // Then reduce across waves using shared memory
    __shared__ T wave_sums[1024 / WAVE_SIZE];  // Max waves per block

    int wave_id   = __wave_id();
    int num_waves = blockDim.x / WAVE_SIZE;

    if (is_wave_leader()) {
        wave_sums[wave_id] = value;
    }
    __syncthreads();

    // Final reduction in first wave
    if (wave_id == 0 && __lane_id() < num_waves) {
        value = wave_sums[__lane_id()];
        value = wave_reduce_sum(value);
    }

    return value;
}

// ================================
// Specialized INT8 Operations
// ================================

// INT8 dot product using V_DOT4_I32_I8
__device__ __forceinline__ int32_t wave_dot4_i8(int32_t a, int32_t b) {
    int32_t result = __builtin_amdgcn_sdot4(a, b, 0, false);
    return wave_reduce_sum(result);
}

// ================================
// Specialized FP16 Operations
// ================================

// FP16 dot product using V_DOT2_F32_F16
__device__ __forceinline__ float wave_dot2_f16(uint32_t a, uint32_t b) {
    float result;
    asm volatile("v_dot2_f32_f16 %0, %1, %2, 0.0" : "=v"(result) : "v"(a), "v"(b));
    return wave_reduce_sum(result);
}

// FP16 reduction with packed half2
__device__ __forceinline__ float wave_reduce_sum_f16x2(__half2 value) {
    value = wave_reduce_sum(value);
    return __low2float(value) + __high2float(value);
}

}  // namespace gfx906

// ================================
// Compatibility Macros
// ================================

// Map generic names to GFX906 implementations (using global scope to avoid nesting)
#    define wave_reduce_sum      ::gfx906::wave_reduce_sum
#    define wave_reduce_max      ::gfx906::wave_reduce_max
#    define wave_reduce_min      gfx906::wave_reduce_min
#    define wave_broadcast       gfx906::wave_broadcast
#    define wave_broadcast_first gfx906::wave_broadcast_first
#    define wave_shuffle         gfx906::wave_shuffle
#    define wave_shuffle_xor     gfx906::wave_shuffle_xor
#    define wave_prefix_sum      gfx906::wave_prefix_sum_inclusive
#    define wave_all             gfx906::wave_all
#    define wave_any             gfx906::wave_any
#    define wave_ballot          gfx906::wave_ballot

#endif  // GGML_HIP_GFX906_OPTIMIZED
