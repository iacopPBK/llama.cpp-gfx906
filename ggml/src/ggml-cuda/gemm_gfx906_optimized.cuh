#pragma once

#ifdef __HIP_PLATFORM_AMD__

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Optimized GEMM kernel for GFX906 with reduced VGPR usage
// Target: < 64 VGPRs for 16 waves/CU occupancy

constexpr int OPT_TILE_M = 128;
constexpr int OPT_TILE_N = 128; 
constexpr int OPT_TILE_K = 32;
constexpr int OPT_THREADS = 256;
constexpr int OPT_THREAD_TILE_M = 4;  // Reduced from 8
constexpr int OPT_THREAD_TILE_N = 4;  // Reduced from 8

// FP16 optimized kernel with aggressive register reduction
__global__ void gemm_f16_gfx906_optimized(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    __half* __restrict__ C,
    const int M, const int N, const int K,
    const float alpha, const float beta
) {
    const int tid = threadIdx.x;
    const int block_row = blockIdx.y * OPT_TILE_M;
    const int block_col = blockIdx.x * OPT_TILE_N;
    
    // Shared memory with padding for bank conflict avoidance
    __shared__ __half lds_tile_a[OPT_TILE_K][OPT_TILE_M + 1];
    __shared__ __half lds_tile_b[OPT_TILE_K][OPT_TILE_N + 1];
    
    // Calculate thread's position in tile (16x16 thread arrangement)
    const int threads_per_row = OPT_TILE_N / OPT_THREAD_TILE_N;  // 32 threads
    const int thread_row = (tid / threads_per_row) * OPT_THREAD_TILE_M;
    const int thread_col = (tid % threads_per_row) * OPT_THREAD_TILE_N;
    
    // Reduced accumulator array - only 4x4 = 16 floats (16 VGPRs)
    float acc[OPT_THREAD_TILE_M][OPT_THREAD_TILE_N];
    #pragma unroll
    for (int i = 0; i < OPT_THREAD_TILE_M; i++) {
        #pragma unroll
        for (int j = 0; j < OPT_THREAD_TILE_N; j++) {
            acc[i][j] = 0.0f;
        }
    }
    
    // Main GEMM loop with double buffering preparation
    for (int k_tile = 0; k_tile < K; k_tile += OPT_TILE_K) {
        // Cooperative loading with vectorized loads
        // Each thread loads 2 half2 values (4 halfs total)
        int load_a_row = tid / (OPT_TILE_K / 2);  // 256 / 16 = 16 rows
        int load_a_col = (tid % (OPT_TILE_K / 2)) * 2;
        
        int load_b_row = tid / (OPT_TILE_N / 2);  // 256 / 64 = 4 rows  
        int load_b_col = (tid % (OPT_TILE_N / 2)) * 2;
        
        // Load A tile (reuse registers aggressively)
        if (load_a_row < OPT_TILE_M && k_tile + load_a_col < K) {
            half2 a_data = *reinterpret_cast<const half2*>(
                &A[(block_row + load_a_row) * K + k_tile + load_a_col]);
            lds_tile_a[load_a_col][load_a_row] = a_data.x;
            lds_tile_a[load_a_col + 1][load_a_row] = a_data.y;
        }
        
        // Load B tile  
        if (load_b_row < OPT_TILE_K && load_b_col < OPT_TILE_N) {
            half2 b_data = *reinterpret_cast<const half2*>(
                &B[(k_tile + load_b_row) * N + block_col + load_b_col]);
            lds_tile_b[load_b_row][load_b_col] = b_data.x;
            lds_tile_b[load_b_row][load_b_col + 1] = b_data.y;
        }
        
        __syncthreads();
        
        // Computation with V_DOT2_F32_F16 optimization
        // Process 2 K values at a time using dot product instruction
        #pragma unroll
        for (int k = 0; k < OPT_TILE_K; k += 2) {
            // Load and pack A values (4 VGPRs for 4x2 values)
            half2 a_packed[OPT_THREAD_TILE_M];
            #pragma unroll
            for (int m = 0; m < OPT_THREAD_TILE_M; m++) {
                a_packed[m] = half2(
                    lds_tile_a[k][thread_row + m],
                    lds_tile_a[k + 1][thread_row + m]
                );
            }
            
            // Load and pack B values (4 VGPRs for 4x2 values)
            half2 b_packed[OPT_THREAD_TILE_N];
            #pragma unroll
            for (int n = 0; n < OPT_THREAD_TILE_N; n++) {
                b_packed[n] = half2(
                    lds_tile_b[k][thread_col + n],
                    lds_tile_b[k + 1][thread_col + n]
                );
            }
            
            // Perform dot products using inline assembly
            #pragma unroll
            for (int m = 0; m < OPT_THREAD_TILE_M; m++) {
                #pragma unroll
                for (int n = 0; n < OPT_THREAD_TILE_N; n++) {
                    // Use V_DOT2_F32_F16 instruction via intrinsic
                    asm volatile(
                        "v_dot2_f32_f16 %0, %1, %2, %0\n\t"
                        : "+v"(acc[m][n])
                        : "v"(*reinterpret_cast<uint32_t*>(&a_packed[m])),
                          "v"(*reinterpret_cast<uint32_t*>(&b_packed[n]))
                    );
                }
            }
        }
        
        __syncthreads();
    }
    
    // Store results
    #pragma unroll
    for (int m = 0; m < OPT_THREAD_TILE_M; m++) {
        #pragma unroll
        for (int n = 0; n < OPT_THREAD_TILE_N; n++) {
            int row = block_row + thread_row + m;
            int col = block_col + thread_col + n;
            if (row < M && col < N) {
                if (beta == 0.0f) {
                    C[row * N + col] = __float2half(alpha * acc[m][n]);
                } else {
                    float c_val = __half2float(C[row * N + col]);
                    C[row * N + col] = __float2half(alpha * acc[m][n] + beta * c_val);
                }
            }
        }
    }
}

// FP32 version with similar optimizations
__global__ void gemm_f32_gfx906_optimized(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int M, const int N, const int K,
    const float alpha, const float beta
) {
    const int tid = threadIdx.x;
    const int block_row = blockIdx.y * OPT_TILE_M;
    const int block_col = blockIdx.x * OPT_TILE_N;
    
    // Shared memory
    __shared__ float lds_tile_a[OPT_TILE_K][OPT_TILE_M + 1];
    __shared__ float lds_tile_b[OPT_TILE_K][OPT_TILE_N + 1];
    
    const int threads_per_row = OPT_TILE_N / OPT_THREAD_TILE_N;
    const int thread_row = (tid / threads_per_row) * OPT_THREAD_TILE_M;
    const int thread_col = (tid % threads_per_row) * OPT_THREAD_TILE_N;
    
    // 4x4 accumulator (16 VGPRs)
    float acc[OPT_THREAD_TILE_M][OPT_THREAD_TILE_N] = {{0.0f}};
    
    // Main loop
    for (int k_tile = 0; k_tile < K; k_tile += OPT_TILE_K) {
        // Cooperative loading with float4 for coalescing
        int load_a_row = tid / (OPT_TILE_K / 4);
        int load_a_col = (tid % (OPT_TILE_K / 4)) * 4;
        
        if (load_a_row < OPT_TILE_M && k_tile + load_a_col < K) {
            float4 a_data = *reinterpret_cast<const float4*>(
                &A[(block_row + load_a_row) * K + k_tile + load_a_col]);
            lds_tile_a[load_a_col][load_a_row] = a_data.x;
            lds_tile_a[load_a_col + 1][load_a_row] = a_data.y;
            lds_tile_a[load_a_col + 2][load_a_row] = a_data.z;
            lds_tile_a[load_a_col + 3][load_a_row] = a_data.w;
        }
        
        int load_b_row = tid / (OPT_TILE_N / 4);
        int load_b_col = (tid % (OPT_TILE_N / 4)) * 4;
        
        if (load_b_row < OPT_TILE_K && load_b_col < OPT_TILE_N) {
            float4 b_data = *reinterpret_cast<const float4*>(
                &B[(k_tile + load_b_row) * N + block_col + load_b_col]);
            lds_tile_b[load_b_row][load_b_col] = b_data.x;
            lds_tile_b[load_b_row][load_b_col + 1] = b_data.y;
            lds_tile_b[load_b_row][load_b_col + 2] = b_data.z;
            lds_tile_b[load_b_row][load_b_col + 3] = b_data.w;
        }
        
        __syncthreads();
        
        // Computation with aggressive unrolling
        #pragma unroll
        for (int k = 0; k < OPT_TILE_K; k++) {
            // Use temporary registers for A and B values
            float a_reg[OPT_THREAD_TILE_M];
            float b_reg[OPT_THREAD_TILE_N];
            
            #pragma unroll
            for (int m = 0; m < OPT_THREAD_TILE_M; m++) {
                a_reg[m] = lds_tile_a[k][thread_row + m];
            }
            
            #pragma unroll
            for (int n = 0; n < OPT_THREAD_TILE_N; n++) {
                b_reg[n] = lds_tile_b[k][thread_col + n];
            }
            
            // FMA operations
            #pragma unroll
            for (int m = 0; m < OPT_THREAD_TILE_M; m++) {
                #pragma unroll
                for (int n = 0; n < OPT_THREAD_TILE_N; n++) {
                    acc[m][n] = fmaf(a_reg[m], b_reg[n], acc[m][n]);
                }
            }
        }
        
        __syncthreads();
    }
    
    // Store results
    #pragma unroll
    for (int m = 0; m < OPT_THREAD_TILE_M; m++) {
        #pragma unroll
        for (int n = 0; n < OPT_THREAD_TILE_N; n++) {
            int row = block_row + thread_row + m;
            int col = block_col + thread_col + n;
            if (row < M && col < N) {
                if (beta == 0.0f) {
                    C[row * N + col] = alpha * acc[m][n];
                } else {
                    C[row * N + col] = alpha * acc[m][n] + beta * C[row * N + col];
                }
            }
        }
    }
}

#endif // __HIP_PLATFORM_AMD__