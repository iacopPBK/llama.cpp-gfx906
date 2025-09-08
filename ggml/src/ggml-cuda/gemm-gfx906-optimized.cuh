#pragma once

#include "common.cuh"
#include "gfx906-config.cuh"

#ifdef GGML_USE_HIP
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_vector_types.h>
#endif

#ifdef GGML_HIP_GFX906_OPTIMIZED

// Optimized GEMM kernel for GFX906 using inline assembly
// Based on AMD ISA documentation and best practices

// Configuration for optimal GFX906 performance
#define GEMM_OPT_TILE_M 128
#define GEMM_OPT_TILE_N 128  
#define GEMM_OPT_TILE_K 32
#define GEMM_OPT_THREADS 256  // 4 wavefronts
#define GEMM_OPT_THREAD_TILE_M 2
#define GEMM_OPT_THREAD_TILE_N 2

// Optimized FP16 GEMM kernel using inline assembly
template<int TILE_M = GEMM_OPT_TILE_M, int TILE_N = GEMM_OPT_TILE_N, int TILE_K = GEMM_OPT_TILE_K>
__global__ void gemm_f16_gfx906_optimized(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    const int M, const int N, const int K,
    const float alpha, const float beta
) {
    // Shared memory for single-buffered tiles (simpler than double buffering)
    __shared__ half lds_tile_a[TILE_M][TILE_K + 1];  // +1 for bank conflict avoidance
    __shared__ half lds_tile_b[TILE_K][TILE_N + 1];
    
    const int tid = threadIdx.x;
    const int block_row = blockIdx.y * TILE_M;
    const int block_col = blockIdx.x * TILE_N;
    
    // Each thread computes a 2x2 tile
    const int threads_per_row = TILE_N / GEMM_OPT_THREAD_TILE_N;
    const int thread_row = (tid / threads_per_row) * GEMM_OPT_THREAD_TILE_M;
    const int thread_col = (tid % threads_per_row) * GEMM_OPT_THREAD_TILE_N;
    
    // Accumulator registers - these stay in VGPRs for entire kernel
    float c_accum[GEMM_OPT_THREAD_TILE_M][GEMM_OPT_THREAD_TILE_N] = {{0.0f}};
    
    // Main loop over K dimension
    for (int k_tile = 0; k_tile < K; k_tile += TILE_K) {
        
        // Cooperative loading of tiles from global to LDS using vectorized loads
        // Each thread loads 128 bits (8 half values) at a time
        const int loads_per_tile_a = (TILE_M * TILE_K) / (GEMM_OPT_THREADS * 8);
        const int loads_per_tile_b = (TILE_K * TILE_N) / (GEMM_OPT_THREADS * 8);
        
        // Load A tile with float4 (8 halfs)
        #pragma unroll
        for (int i = 0; i < loads_per_tile_a; i++) {
            const int idx = tid + i * GEMM_OPT_THREADS;
            const int row = (idx * 8) / TILE_K;
            const int col = (idx * 8) % TILE_K;
            
            if (block_row + row < M && k_tile + col + 7 < K) {
                // Load 128 bits at once
                float4 data = *reinterpret_cast<const float4*>(&A[(block_row + row) * K + k_tile + col]);
                *reinterpret_cast<float4*>(&lds_tile_a[row][col]) = data;
            } else {
                // Handle boundary
                for (int j = 0; j < 8; j++) {
                    if (block_row + row < M && k_tile + col + j < K) {
                        lds_tile_a[row][col + j] = A[(block_row + row) * K + k_tile + col + j];
                    } else {
                        lds_tile_a[row][col + j] = __float2half(0.0f);
                    }
                }
            }
        }
        
        // Load B tile with float4 (8 halfs)
        #pragma unroll
        for (int i = 0; i < loads_per_tile_b; i++) {
            const int idx = tid + i * GEMM_OPT_THREADS;
            const int row = (idx * 8) / TILE_N;
            const int col = (idx * 8) % TILE_N;
            
            if (k_tile + row < K && block_col + col + 7 < N) {
                // Load 128 bits at once
                float4 data = *reinterpret_cast<const float4*>(&B[(k_tile + row) * N + block_col + col]);
                *reinterpret_cast<float4*>(&lds_tile_b[row][col]) = data;
            } else {
                // Handle boundary
                for (int j = 0; j < 8; j++) {
                    if (k_tile + row < K && block_col + col + j < N) {
                        lds_tile_b[row][col + j] = B[(k_tile + row) * N + block_col + col + j];
                    } else {
                        lds_tile_b[row][col + j] = __float2half(0.0f);
                    }
                }
            }
        }
        
        // Synchronize before computation
        __syncthreads();
        
        // Core computation loop with inline assembly
        // Process 2 K values at a time for V_DOT2_F32_F16
        #pragma unroll
        for (int k = 0; k < TILE_K; k += 2) {
            // Load packed f16 data from LDS
            // Each uint32_t holds two half-precision floats
            uint32_t a_packed[GEMM_OPT_THREAD_TILE_M];
            uint32_t b_packed[GEMM_OPT_THREAD_TILE_N];
            
            // Load A values (2 consecutive K values per row)
            #pragma unroll
            for (int m = 0; m < GEMM_OPT_THREAD_TILE_M; m++) {
                a_packed[m] = *reinterpret_cast<const uint32_t*>(&lds_tile_a[thread_row + m][k]);
            }
            
            // Load B values (2 consecutive K values per column)
            #pragma unroll
            for (int n = 0; n < GEMM_OPT_THREAD_TILE_N; n++) {
                // Pack two consecutive K values for each N  
                const half k0 = lds_tile_b[k][thread_col + n];
                const half k1 = lds_tile_b[k + 1][thread_col + n];
                const half2 h2 = half2(k0, k1);
                b_packed[n] = *reinterpret_cast<const uint32_t*>(&h2);
            }
            
            // Inline assembly for maximum performance
            // Using V_DOT2_F32_F16 instruction directly
            asm volatile(
                // Wait for all LDS reads to complete
                "s_waitcnt lgkmcnt(0)\n\t"
                
                // Compute all 2x2 dot products
                // C[0][0] += A[0][k:k+1] * B[k:k+1][0]
                "v_dot2_f32_f16 %0, %4, %6, %0\n\t"
                
                // C[0][1] += A[0][k:k+1] * B[k:k+1][1]
                "v_dot2_f32_f16 %1, %4, %7, %1\n\t"
                
                // C[1][0] += A[1][k:k+1] * B[k:k+1][0]
                "v_dot2_f32_f16 %2, %5, %6, %2\n\t"
                
                // C[1][1] += A[1][k:k+1] * B[k:k+1][1]
                "v_dot2_f32_f16 %3, %5, %7, %3\n\t"
                
                // Output operands (read-write accumulators)
                : "+v"(c_accum[0][0]), // %0
                  "+v"(c_accum[0][1]), // %1
                  "+v"(c_accum[1][0]), // %2
                  "+v"(c_accum[1][1])  // %3
                
                // Input operands
                : "v"(a_packed[0]),    // %4
                  "v"(a_packed[1]),    // %5
                  "v"(b_packed[0]),    // %6
                  "v"(b_packed[1])     // %7
            );
        }
        
        // Synchronize before next tile load
        __syncthreads();
    }
    
    // Store results using vectorized stores
    const int global_row = block_row + thread_row;
    const int global_col = block_col + thread_col;
    
    // Convert accumulators to half and store
    #pragma unroll
    for (int m = 0; m < GEMM_OPT_THREAD_TILE_M; m++) {
        if (global_row + m < M) {
            #pragma unroll
            for (int n = 0; n < GEMM_OPT_THREAD_TILE_N; n++) {
                if (global_col + n < N) {
                    const int idx = (global_row + m) * N + (global_col + n);
                    
                    // Apply alpha and beta
                    float result = alpha * c_accum[m][n];
                    if (beta != 0.0f) {
                        result += beta * __half2float(C[idx]);
                    }
                    
                    C[idx] = __float2half(result);
                }
            }
        }
    }
}

// Optimized FP32 GEMM kernel using inline assembly patterns
template<int TILE_M = GEMM_OPT_TILE_M, int TILE_N = GEMM_OPT_TILE_N, int TILE_K = GEMM_OPT_TILE_K>
__global__ void gemm_f32_gfx906_optimized(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int M, const int N, const int K,
    const float alpha, const float beta
) {
    // Shared memory for single-buffered tiles
    __shared__ float lds_tile_a[TILE_M][TILE_K + 1];  // +1 for bank conflict avoidance
    __shared__ float lds_tile_b[TILE_K][TILE_N + 1];
    
    const int tid = threadIdx.x;
    const int block_row = blockIdx.y * TILE_M;
    const int block_col = blockIdx.x * TILE_N;
    
    // Each thread computes a 4x4 tile for FP32
    const int THREAD_TILE_M = 4;
    const int THREAD_TILE_N = 4;
    const int threads_per_row = TILE_N / THREAD_TILE_N;
    const int thread_row = (tid / threads_per_row) * THREAD_TILE_M;
    const int thread_col = (tid % threads_per_row) * THREAD_TILE_N;
    
    // Accumulator registers
    float c_accum[THREAD_TILE_M][THREAD_TILE_N] = {{0.0f}};
    
    // Main loop over K dimension
    for (int k_tile = 0; k_tile < K; k_tile += TILE_K) {
        
        // Cooperative loading with float4 for coalescing
        const int float4_per_tile_a = (TILE_M * TILE_K) / (GEMM_OPT_THREADS * 4);
        const int float4_per_tile_b = (TILE_K * TILE_N) / (GEMM_OPT_THREADS * 4);
        
        // Load A tile
        #pragma unroll
        for (int i = 0; i < float4_per_tile_a; i++) {
            const int idx = tid + i * GEMM_OPT_THREADS;
            const int row = (idx * 4) / TILE_K;
            const int col = (idx * 4) % TILE_K;
            
            if (block_row + row < M && k_tile + col + 3 < K) {
                float4 data = *reinterpret_cast<const float4*>(&A[(block_row + row) * K + k_tile + col]);
                lds_tile_a[row][col] = data.x;
                lds_tile_a[row][col + 1] = data.y;
                lds_tile_a[row][col + 2] = data.z;
                lds_tile_a[row][col + 3] = data.w;
            }
        }
        
        // Load B tile
        #pragma unroll
        for (int i = 0; i < float4_per_tile_b; i++) {
            const int idx = tid + i * GEMM_OPT_THREADS;
            const int row = (idx * 4) / TILE_N;
            const int col = (idx * 4) % TILE_N;
            
            if (k_tile + row < K && block_col + col + 3 < N) {
                float4 data = *reinterpret_cast<const float4*>(&B[(k_tile + row) * N + block_col + col]);
                lds_tile_b[row][col] = data.x;
                lds_tile_b[row][col + 1] = data.y;
                lds_tile_b[row][col + 2] = data.z;
                lds_tile_b[row][col + 3] = data.w;
            }
        }
        
        __syncthreads();
        
        // Core computation loop with explicit FMA instructions
        #pragma unroll
        for (int k = 0; k < TILE_K; k++) {
            // Load values from LDS
            float a_vals[THREAD_TILE_M];
            float b_vals[THREAD_TILE_N];
            
            #pragma unroll
            for (int m = 0; m < THREAD_TILE_M; m++) {
                a_vals[m] = lds_tile_a[thread_row + m][k];
            }
            
            #pragma unroll
            for (int n = 0; n < THREAD_TILE_N; n++) {
                b_vals[n] = lds_tile_b[k][thread_col + n];
            }
            
            // Use inline assembly for FMA operations
            #pragma unroll
            for (int m = 0; m < THREAD_TILE_M; m++) {
                #pragma unroll
                for (int n = 0; n < THREAD_TILE_N; n++) {
                    asm volatile(
                        "v_fma_f32 %0, %1, %2, %0"
                        : "+v"(c_accum[m][n])
                        : "v"(a_vals[m]), "v"(b_vals[n])
                    );
                }
            }
        }
        
        __syncthreads();
    }
    
    // Store results with vectorized stores
    const int global_row = block_row + thread_row;
    const int global_col = block_col + thread_col;
    
    #pragma unroll
    for (int m = 0; m < THREAD_TILE_M; m++) {
        if (global_row + m < M && global_col + 3 < N) {
            // Pack 4 results and store with float4
            float4 results;
            results.x = alpha * c_accum[m][0] + beta * C[(global_row + m) * N + global_col];
            results.y = alpha * c_accum[m][1] + beta * C[(global_row + m) * N + global_col + 1];
            results.z = alpha * c_accum[m][2] + beta * C[(global_row + m) * N + global_col + 2];
            results.w = alpha * c_accum[m][3] + beta * C[(global_row + m) * N + global_col + 3];
            
            *reinterpret_cast<float4*>(&C[(global_row + m) * N + global_col]) = results;
        } else {
            // Handle boundary with scalar stores
            for (int n = 0; n < THREAD_TILE_N; n++) {
                if (global_row + m < M && global_col + n < N) {
                    const int idx = (global_row + m) * N + (global_col + n);
                    C[idx] = alpha * c_accum[m][n] + beta * C[idx];
                }
            }
        }
    }
}

// Launcher functions
inline void launch_gemm_f16_gfx906_optimized(
    const half* A, const half* B, half* C,
    const int M, const int N, const int K,
    const float alpha, const float beta,
    hipStream_t stream
) {
    dim3 grid((N + GEMM_OPT_TILE_N - 1) / GEMM_OPT_TILE_N,
              (M + GEMM_OPT_TILE_M - 1) / GEMM_OPT_TILE_M);
    dim3 block(GEMM_OPT_THREADS);
    
    // Calculate shared memory size
    const size_t smem_size = sizeof(half) * (GEMM_OPT_TILE_M * (GEMM_OPT_TILE_K + 1) +
                                             GEMM_OPT_TILE_K * (GEMM_OPT_TILE_N + 1));
    
    gemm_f16_gfx906_optimized<<<grid, block, smem_size, stream>>>(
        A, B, C, M, N, K, alpha, beta
    );
}

inline void launch_gemm_f32_gfx906_optimized(
    const float* A, const float* B, float* C,
    const int M, const int N, const int K,
    const float alpha, const float beta,
    hipStream_t stream
) {
    dim3 grid((N + GEMM_OPT_TILE_N - 1) / GEMM_OPT_TILE_N,
              (M + GEMM_OPT_TILE_M - 1) / GEMM_OPT_TILE_M);
    dim3 block(GEMM_OPT_THREADS);
    
    // Calculate shared memory size
    const size_t smem_size = sizeof(float) * (GEMM_OPT_TILE_M * (GEMM_OPT_TILE_K + 1) +
                                              GEMM_OPT_TILE_K * (GEMM_OPT_TILE_N + 1));
    
    gemm_f32_gfx906_optimized<<<grid, block, smem_size, stream>>>(
        A, B, C, M, N, K, alpha, beta
    );
}

#endif // GGML_HIP_GFX906_OPTIMIZED