#include "common.cuh"
#include "gemm-gfx906.cuh"

#ifdef GGML_HIP_GFX906_OPTIMIZED

// Implementation of GFX906-optimized GEMM operations
// This file provides the main entry points for the optimized GEMM kernels

void ggml_cuda_gemm_gfx906_f32(const float * A,
                               const float * B,
                               float *       C,
                               const int     M,
                               const int     N,
                               const int     K,
                               const float   alpha,
                               const float   beta,
                               cudaStream_t  stream) {
    // Dispatch to the optimized kernel
    launch_gemm_f32_gfx906(A, B, C, M, N, K, alpha, beta, stream);
}

void ggml_cuda_gemm_gfx906_f16(const half * A,
                               const half * B,
                               half *       C,
                               const int    M,
                               const int    N,
                               const int    K,
                               const float  alpha,
                               const float  beta,
                               cudaStream_t stream) {
    // Dispatch to the optimized kernel
    launch_gemm_f16_gfx906(A, B, C, M, N, K, alpha, beta, stream);
}

// Helper function to check if GFX906 optimized GEMM should be used
bool should_use_gfx906_gemm(const int M, const int N, const int K) {
    // Use optimized kernel for sufficiently large matrices
    // Small matrices may not benefit from the overhead
    const int min_size = 256;
    return (M >= min_size && N >= min_size && K >= min_size);
}

// Performance tuning function
gfx906_perf_config get_gemm_perf_config(const int M, const int N, const int K) {
    gfx906_perf_config config;

    // Adjust configuration based on matrix dimensions
    if (M * N * K >= 1000000000) {  // Very large matrices
        config.block_size   = 256;
        config.waves_per_cu = 8;
        config.lds_usage    = 65536;      // Use full LDS
    } else if (M * N * K >= 100000000) {  // Large matrices
        config.block_size   = 256;
        config.waves_per_cu = 4;
        config.lds_usage    = 49152;  // 48KB LDS
    } else {                          // Medium matrices
        config.block_size   = 128;
        config.waves_per_cu = 2;
        config.lds_usage    = 32768;  // 32KB LDS
    }

    config.grid_size = gfx906_get_optimal_grid_size(M * N, config.block_size);

    return config;
}

// Export functions for testing
extern "C" {
    bool ggml_cuda_gemm_gfx906_available() {
        return true;
    }
}

#endif  // GGML_HIP_GFX906_OPTIMIZED
