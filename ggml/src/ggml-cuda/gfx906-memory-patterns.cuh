#pragma once

// GFX906 Optimized Memory Access Patterns
// Implements vectorized 128-bit aligned loads, coalesced global memory access,
// and LDS double buffering for AMD Instinct MI50 (GFX906) to achieve 900+ GB/s
// sustained HBM2 bandwidth

#ifdef GGML_HIP_GFX906_OPTIMIZED

#    include "gfx906-config.cuh"

#    include <hip/hip_runtime.h>

namespace gfx906 {
namespace memory {

// ================================
// Constants and Configuration
// ================================

// HBM2 memory alignment requirements
static constexpr size_t HBM2_ALIGNMENT   = 256;  // 256-byte alignment for optimal HBM2 access
static constexpr size_t CACHE_LINE_SIZE  = 64;   // 64-byte cache line
static constexpr size_t VECTOR_LOAD_SIZE = 16;   // 128-bit (16-byte) vectorized loads

// LDS double buffering configuration
static constexpr size_t LDS_BANK_SIZE = 4;   // 4-byte banks
static constexpr int    LDS_NUM_BANKS = 32;  // 32 banks in GFX906
static constexpr size_t LDS_PADDING   = 4;   // Padding to avoid bank conflicts

// ================================
// Vectorized 128-bit Aligned Loads
// ================================

// Load 128-bit (float4) with alignment check
__device__ __forceinline__ float4 load_float4_aligned(const float * __restrict__ addr) {
    // Ensure 16-byte alignment for 128-bit load
    assert(((uintptr_t) addr & 0xF) == 0);
    return *reinterpret_cast<const float4 *>(addr);
}

// Load 128-bit (int4) with alignment check
__device__ __forceinline__ int4 load_int4_aligned(const int * __restrict__ addr) {
    assert(((uintptr_t) addr & 0xF) == 0);
    return *reinterpret_cast<const int4 *>(addr);
}

// Vectorized load for half precision (8 x half = 128 bits)
__device__ __forceinline__ float4 load_half8_as_float4(const __half * __restrict__ addr) {
    assert(((uintptr_t) addr & 0xF) == 0);
    return *reinterpret_cast<const float4 *>(addr);
}

// Generic vectorized load with automatic vectorization
template <typename T>
__device__ __forceinline__ void load_vectorized_128(T * __restrict__ dst, const T * __restrict__ src, int count) {
    const int vec4_count = count / 4;
    const int remainder  = count % 4;

// Process 128-bit chunks
#    pragma unroll 4
    for (int i = 0; i < vec4_count; i++) {
        float4 vec                               = load_float4_aligned(reinterpret_cast<const float *>(src + i * 4));
        *reinterpret_cast<float4 *>(dst + i * 4) = vec;
    }

    // Handle remainder
    for (int i = vec4_count * 4; i < count; i++) {
        dst[i] = src[i];
    }
}

// ================================
// Coalesced Global Memory Access
// ================================

// Coalesced load pattern for warp-wide access
template <typename T, int BLOCK_SIZE>
__device__ __forceinline__ void coalesced_load(T * __restrict__ dst, const T * __restrict__ src, int n_elements) {
    const int tid     = threadIdx.x;
    const int lane_id = tid % warpSize;
    const int warp_id = tid / warpSize;
    const int n_warps = BLOCK_SIZE / warpSize;

    // Each warp loads contiguous 128-byte chunks for coalescing
    const int elements_per_warp = warpSize;
    const int warp_offset       = warp_id * elements_per_warp;

    // Coalesced access pattern
    for (int i = warp_offset + lane_id; i < n_elements; i += n_warps * elements_per_warp) {
        dst[i] = src[i];
    }
}

// Strided coalesced load for matrix access
template <typename T>
__device__ __forceinline__ void coalesced_matrix_load(T * __restrict__ dst,
                                                      const T * __restrict__ src,
                                                      int rows,
                                                      int cols,
                                                      int src_stride) {
    const int tid        = threadIdx.x;
    const int block_size = blockDim.x;

    // Coalesce across columns (contiguous in memory)
    for (int row = 0; row < rows; row++) {
        for (int col = tid; col < cols; col += block_size) {
            dst[row * cols + col] = src[row * src_stride + col];
        }
    }
}

// ================================
// LDS Double Buffering
// ================================

template <typename T, int TILE_SIZE_M, int TILE_SIZE_N> class LDSDoubleBuffer {
  private:
    // Double buffer with padding to avoid bank conflicts
    __shared__ T buffer[2][TILE_SIZE_M][TILE_SIZE_N + LDS_PADDING];
    int          current_buffer;

  public:
    __device__ LDSDoubleBuffer() : current_buffer(0) {}

    // Get pointer to current read buffer
    __device__ __forceinline__ T * get_read_buffer() { return &buffer[current_buffer][0][0]; }

    // Get pointer to current write buffer
    __device__ __forceinline__ T * get_write_buffer() { return &buffer[1 - current_buffer][0][0]; }

    // Swap buffers
    __device__ __forceinline__ void swap() { current_buffer = 1 - current_buffer; }

    // Async load from global to LDS (write buffer)
    __device__ __forceinline__ void async_load_tile(const T * __restrict__ gmem_src,
                                                    int gmem_stride,
                                                    int tile_row_start,
                                                    int tile_col_start) {
        const int tid        = threadIdx.x;
        const int block_size = blockDim.x;
        T *       write_buf  = get_write_buffer();

        // Cooperative tile loading with coalesced access
        const int elements_per_row = TILE_SIZE_N;

        for (int row = 0; row < TILE_SIZE_M; row++) {
            for (int col = tid; col < TILE_SIZE_N; col += block_size) {
                int gmem_row                                       = tile_row_start + row;
                int gmem_col                                       = tile_col_start + col;
                write_buf[row * (TILE_SIZE_N + LDS_PADDING) + col] = gmem_src[gmem_row * gmem_stride + gmem_col];
            }
        }
    }

    // Synchronize and swap buffers
    __device__ __forceinline__ void sync_and_swap() {
        __syncthreads();
        swap();
    }
};

// ================================
// Prefetching Strategies
// ================================

// Software prefetch for next iteration
template <typename T> __device__ __forceinline__ void prefetch_l2(const T * addr) {
    // GFX906 specific prefetch hint
    // Using __builtin_prefetch equivalent for HIP
    asm volatile(
        "global_load_dword v[0:1], %0, off glc slc\n\t"
        "s_waitcnt vmcnt(0)"
        :
        : "v"(addr)
        : "memory");
}

// Prefetch with distance control
template <typename T, int PREFETCH_DISTANCE>
__device__ __forceinline__ void prefetch_ahead(const T * __restrict__ base_addr, int current_idx, int max_idx) {
    int prefetch_idx = current_idx + PREFETCH_DISTANCE;
    if (prefetch_idx < max_idx) {
        prefetch_l2(&base_addr[prefetch_idx]);
    }
}

// ================================
// Memory Bandwidth Optimization
// ================================

// Optimized memory copy for maximum bandwidth utilization
template <typename T>
__global__ void optimized_memcpy_kernel(T * __restrict__ dst, const T * __restrict__ src, size_t n_elements) {
    const int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    // Use float4 for 128-bit loads/stores
    const size_t   n_float4 = n_elements * sizeof(T) / sizeof(float4);
    float4 *       dst_f4   = reinterpret_cast<float4 *>(dst);
    const float4 * src_f4   = reinterpret_cast<const float4 *>(src);

    // Vectorized copy with coalesced access
    for (size_t i = tid; i < n_float4; i += stride) {
        dst_f4[i] = src_f4[i];
    }

    // Handle remainder bytes
    const size_t remainder_start = n_float4 * sizeof(float4) / sizeof(T);
    if (tid == 0) {
        for (size_t i = remainder_start; i < n_elements; i++) {
            dst[i] = src[i];
        }
    }
}

// Launch configuration for optimal bandwidth
inline void get_optimal_memcpy_config(size_t n_elements, int & grid_size, int & block_size) {
    // Optimize for GFX906: 60 CUs, 1024 max threads per block
    block_size = 256;  // 4 waves per block

    // Calculate grid to saturate memory bandwidth
    const size_t elements_per_block = block_size * 4;  // float4 per thread
    grid_size                       = (n_elements + elements_per_block - 1) / elements_per_block;

    // Cap at maximum CU utilization
    const int max_blocks = GFX906_NUM_CUS * 16;  // 16 blocks per CU
    grid_size            = min(grid_size, max_blocks);
}

// ================================
// Memory Access Pattern Utilities
// ================================

// Check and enforce alignment
template <typename T> __device__ __forceinline__ bool is_aligned(const T * ptr, size_t alignment) {
    return ((uintptr_t) ptr & (alignment - 1)) == 0;
}

// Align pointer to boundary
template <typename T> __device__ __forceinline__ T * align_ptr(T * ptr, size_t alignment) {
    uintptr_t addr    = (uintptr_t) ptr;
    uintptr_t aligned = (addr + alignment - 1) & ~(alignment - 1);
    return reinterpret_cast<T *>(aligned);
}

// Calculate padding for alignment
__device__ __forceinline__ size_t calculate_padding(size_t size, size_t alignment) {
    return (alignment - (size & (alignment - 1))) & (alignment - 1);
}

// ================================
// Bandwidth Monitoring
// ================================

// Structure for bandwidth statistics
struct BandwidthStats {
    float  achieved_bandwidth_gbps;
    float  efficiency_percent;
    size_t bytes_transferred;
    float  elapsed_time_ms;

    __host__ void print() const {
        printf("Memory Bandwidth Statistics:\n");
        printf("  Achieved: %.2f GB/s\n", achieved_bandwidth_gbps);
        printf("  Efficiency: %.1f%% of theoretical peak\n", efficiency_percent);
        printf("  Transferred: %.2f GB in %.2f ms\n", bytes_transferred / 1e9, elapsed_time_ms);
    }
};

// Calculate bandwidth from timing
__host__ inline BandwidthStats calculate_bandwidth(size_t bytes_transferred, float elapsed_time_ms) {
    BandwidthStats stats;
    stats.bytes_transferred       = bytes_transferred;
    stats.elapsed_time_ms         = elapsed_time_ms;
    stats.achieved_bandwidth_gbps = (bytes_transferred / 1e9) / (elapsed_time_ms / 1e3);
    stats.efficiency_percent      = (stats.achieved_bandwidth_gbps / GFX906_HBM2_BANDWIDTH) * 100.0f;
    return stats;
}

}  // namespace memory
}  // namespace gfx906

#endif  // GGML_HIP_GFX906_OPTIMIZED
