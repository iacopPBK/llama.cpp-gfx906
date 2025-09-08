#pragma once

// GFX906 (AMD Instinct MI50) Backend Infrastructure Configuration
// This file contains hardware-specific configurations for the GFX906 architecture

#ifdef GGML_HIP_GFX906_OPTIMIZED

// Hardware specifications for GFX906 (AMD Instinct MI50)
#    define GFX906_NUM_CUS               60     // Number of Compute Units
#    define GFX906_LDS_SIZE              65536  // 64KB Local Data Share per CU
#    define GFX906_WAVE_SIZE             64     // Wave size for GCN architecture
#    define GFX906_MAX_THREADS_PER_BLOCK 1024
#    define GFX906_MAX_BLOCKS_PER_CU     40     // Maximum concurrent blocks per CU
#    define GFX906_SIMD_WIDTH            64     // SIMD width

// Memory hierarchy configuration
#    define GFX906_L1_CACHE_SIZE  16384    // 16KB L1 cache per CU
#    define GFX906_L2_CACHE_SIZE  4194304  // 4MB L2 cache total
#    define GFX906_HBM2_BANDWIDTH 1024     // GB/s theoretical peak

// Hardware capabilities for GFX906
#    define GFX906_HAS_DP4A      1  // Support for dot product instruction
#    define GFX906_HAS_FP16      1  // Hardware FP16 support
#    define GFX906_HAS_INT8      1  // INT8 support
#    define GFX906_HAS_DUAL_FP16 1  // Dual FP16 issue capability

// Stream management configuration
#    define GFX906_MAX_STREAMS     16  // Maximum concurrent streams
#    define GFX906_DEFAULT_STREAMS 4   // Default number of streams

// Kernel launch configuration helpers
static inline dim3 gfx906_get_optimal_block_size(int n) {
    // Optimize for wave size and occupancy
    if (n <= 256) {
        return dim3(256, 1, 1);   // 4 waves per block
    } else if (n <= 512) {
        return dim3(512, 1, 1);   // 8 waves per block
    } else {
        return dim3(1024, 1, 1);  // 16 waves per block (max)
    }
}

static inline int gfx906_get_optimal_grid_size(int n, int block_size) {
    // Calculate grid size based on CU count and maximum blocks per CU
    const int max_blocks    = GFX906_NUM_CUS * GFX906_MAX_BLOCKS_PER_CU;
    const int needed_blocks = (n + block_size - 1) / block_size;
    return (needed_blocks < max_blocks) ? needed_blocks : max_blocks;
}

// LDS (Local Data Share) allocation helpers
static inline size_t gfx906_get_max_lds_per_block() {
    return GFX906_LDS_SIZE;  // 64KB max per block
}

static inline size_t gfx906_calculate_lds_usage(int threads_per_block, size_t per_thread_lds) {
    // Ensure we don't exceed LDS limits
    size_t total_lds = threads_per_block * per_thread_lds;
    return (total_lds <= GFX906_LDS_SIZE) ? total_lds : 0;
}

// Device detection helper
static inline bool is_gfx906_device(int device_cc) {
    // Check if the device compute capability matches GFX906
    return (device_cc & 0xffff) == 0x906;
}

// Performance tuning parameters
struct gfx906_perf_config {
    int    block_size;
    int    grid_size;
    size_t lds_usage;
    int    waves_per_cu;

    gfx906_perf_config() : block_size(256), grid_size(GFX906_NUM_CUS * 4), lds_usage(16384), waves_per_cu(4) {}
};

// Get optimal configuration for different operation types
static inline gfx906_perf_config gfx906_get_gemm_config(int m, int n, int k) {
    gfx906_perf_config config;

    // Tune for matrix multiplication on GFX906
    if (k >= 2048) {
        config.block_size   = 256;
        config.waves_per_cu = 8;
    } else if (k >= 1024) {
        config.block_size   = 128;
        config.waves_per_cu = 4;
    } else {
        config.block_size   = 64;
        config.waves_per_cu = 2;
    }

    config.grid_size = gfx906_get_optimal_grid_size(m * n, config.block_size);
    config.lds_usage = config.block_size * sizeof(float) * 32;  // Example LDS usage

    return config;
}

// Memory management helpers
static inline size_t gfx906_align_memory(size_t size) {
    // Align to 256 bytes for optimal HBM2 access
    const size_t alignment = 256;
    return ((size + alignment - 1) / alignment) * alignment;
}

// Compiler hints for GFX906 optimization
#    define GFX906_UNROLL_FACTOR     4
#    define GFX906_PREFETCH_DISTANCE 512

// Enable specific optimizations for GFX906
#    ifdef GGML_USE_HIP
#        define GFX906_USE_FAST_MATH  1
#        define GFX906_USE_INLINE_ASM 1

// Hardware-specific intrinsics
// V_DOT4_I32_I8 instruction for INT8 dot products
__device__ __forceinline__ int32_t gfx906_dot4_i8(int32_t a, int32_t b) {
#        if defined(__HIP_DEVICE_COMPILE__)
    int32_t result;
    // Using HIP intrinsic for dot product
    asm volatile("v_dot4_i32_i8 %0, %1, %2, 0" : "=v"(result) : "v"(a), "v"(b));
    return result;
#        else
    return 0; // Placeholder for host compilation
#        endif
}

// V_DOT2_F32_F16 instruction for FP16 dot products
__device__ __forceinline__ float gfx906_dot2_f16(uint32_t a, uint32_t b, float c) {
#        if defined(__HIP_DEVICE_COMPILE__)
    float result;
    asm volatile("v_dot2_f32_f16 %0, %1, %2, %3" : "=v"(result) : "v"(a), "v"(b), "v"(c));
    return result;
#        else
    return 0.0f; // Placeholder for host compilation
#        endif
}
#    endif

// Debug and profiling helpers
#    ifdef GFX906_ENABLE_PROFILING
#        define GFX906_PROFILE_START(name)        \
            hipEvent_t start_##name, stop_##name; \
            hipEventCreate(&start_##name);        \
            hipEventCreate(&stop_##name);         \
            hipEventRecord(start_##name);

#        define GFX906_PROFILE_END(name)                                  \
            hipEventRecord(stop_##name);                                  \
            hipEventSynchronize(stop_##name);                             \
            float time_##name;                                            \
            hipEventElapsedTime(&time_##name, start_##name, stop_##name); \
            printf("GFX906 Profile [%s]: %.3f ms\n", #name, time_##name); \
            hipEventDestroy(start_##name);                                \
            hipEventDestroy(stop_##name);
#    else
#        define GFX906_PROFILE_START(name)
#        define GFX906_PROFILE_END(name)
#    endif

#endif  // GGML_HIP_GFX906_OPTIMIZED
