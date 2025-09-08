#pragma once

#ifdef __HIP_PLATFORM_AMD__

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Double-buffered GEMM kernel for GFX906
// Overlaps memory loading with computation for latency hiding

// Reduced tile sizes to fit double buffer in 64KB LDS
// FP16: 2 * (16*64 + 16*64) * 2 bytes = 8KB
// FP32: 2 * (16*64 + 16*64) * 4 bytes = 16KB
constexpr int DB_TILE_M = 64;
constexpr int DB_TILE_N = 64;
constexpr int DB_TILE_K = 16;
constexpr int DB_THREADS = 256;
constexpr int DB_THREAD_TILE_M = 4;
constexpr int DB_THREAD_TILE_N = 4;

__global__ void gemm_f16_gfx906_double_buffer(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    __half* __restrict__ C,
    const int M, const int N, const int K,
    const float alpha, const float beta
) {
    const int tid = threadIdx.x;
    const int block_row = blockIdx.y * DB_TILE_M;
    const int block_col = blockIdx.x * DB_TILE_N;
    
    // Double-buffered shared memory (ping-pong buffers)
    __shared__ __half lds_tile_a[2][DB_TILE_K][DB_TILE_M + 1];
    __shared__ __half lds_tile_b[2][DB_TILE_K][DB_TILE_N + 1];
    
    // Thread's position in tile
    const int threads_per_row = DB_TILE_N / DB_THREAD_TILE_N;
    const int thread_row = (tid / threads_per_row) * DB_THREAD_TILE_M;
    const int thread_col = (tid % threads_per_row) * DB_THREAD_TILE_N;
    
    // Accumulator registers (16 VGPRs)
    float acc[DB_THREAD_TILE_M][DB_THREAD_TILE_N] = {{0.0f}};
    
    // Load indices for cooperative loading
    const int load_a_row = tid / (DB_TILE_K / 2);
    const int load_a_col = (tid % (DB_TILE_K / 2)) * 2;
    const int load_b_row = tid / (DB_TILE_N / 2);
    const int load_b_col = (tid % (DB_TILE_N / 2)) * 2;
    
    // Prefetch first tile into buffer 0
    if (load_a_row < DB_TILE_M && load_a_col < K) {
        half2 a_data = *reinterpret_cast<const half2*>(
            &A[(block_row + load_a_row) * K + load_a_col]);
        lds_tile_a[0][load_a_col][load_a_row] = a_data.x;
        lds_tile_a[0][load_a_col + 1][load_a_row] = a_data.y;
    }
    
    if (load_b_row < DB_TILE_K && load_b_col < DB_TILE_N) {
        half2 b_data = *reinterpret_cast<const half2*>(
            &B[load_b_row * N + block_col + load_b_col]);
        lds_tile_b[0][load_b_row][load_b_col] = b_data.x;
        lds_tile_b[0][load_b_row][load_b_col + 1] = b_data.y;
    }
    
    __syncthreads();
    
    // Main GEMM loop with double buffering
    int buffer_idx = 0;
    
    for (int k_tile = 0; k_tile < K; k_tile += DB_TILE_K) {
        int next_buffer = 1 - buffer_idx;
        int next_k_tile = k_tile + DB_TILE_K;
        
        // Asynchronously load next tile while computing current
        if (next_k_tile < K) {
            // Load A tile for next iteration
            if (load_a_row < DB_TILE_M && next_k_tile + load_a_col < K) {
                half2 a_data = *reinterpret_cast<const half2*>(
                    &A[(block_row + load_a_row) * K + next_k_tile + load_a_col]);
                lds_tile_a[next_buffer][load_a_col][load_a_row] = a_data.x;
                lds_tile_a[next_buffer][load_a_col + 1][load_a_row] = a_data.y;
            }
            
            // Load B tile for next iteration
            if (load_b_row < DB_TILE_K && load_b_col < DB_TILE_N) {
                half2 b_data = *reinterpret_cast<const half2*>(
                    &B[(next_k_tile + load_b_row) * N + block_col + load_b_col]);
                lds_tile_b[next_buffer][load_b_row][load_b_col] = b_data.x;
                lds_tile_b[next_buffer][load_b_row][load_b_col + 1] = b_data.y;
            }
        }
        
        // Compute using current buffer with V_DOT2 optimization
        #pragma unroll
        for (int k = 0; k < DB_TILE_K; k += 2) {
            // Load and pack A values
            half2 a_packed[DB_THREAD_TILE_M];
            #pragma unroll
            for (int m = 0; m < DB_THREAD_TILE_M; m++) {
                a_packed[m] = half2(
                    lds_tile_a[buffer_idx][k][thread_row + m],
                    lds_tile_a[buffer_idx][k + 1][thread_row + m]
                );
            }
            
            // Load and pack B values
            half2 b_packed[DB_THREAD_TILE_N];
            #pragma unroll
            for (int n = 0; n < DB_THREAD_TILE_N; n++) {
                b_packed[n] = half2(
                    lds_tile_b[buffer_idx][k][thread_col + n],
                    lds_tile_b[buffer_idx][k + 1][thread_col + n]
                );
            }
            
            // Perform dot products
            #pragma unroll
            for (int m = 0; m < DB_THREAD_TILE_M; m++) {
                #pragma unroll
                for (int n = 0; n < DB_THREAD_TILE_N; n++) {
                    asm volatile(
                        "v_dot2_f32_f16 %0, %1, %2, %0\n\t"
                        : "+v"(acc[m][n])
                        : "v"(*reinterpret_cast<uint32_t*>(&a_packed[m])),
                          "v"(*reinterpret_cast<uint32_t*>(&b_packed[n]))
                    );
                }
            }
        }
        
        // Switch buffers
        buffer_idx = next_buffer;
        __syncthreads();
    }
    
    // Store results
    #pragma unroll
    for (int m = 0; m < DB_THREAD_TILE_M; m++) {
        #pragma unroll
        for (int n = 0; n < DB_THREAD_TILE_N; n++) {
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

// FP32 version with double buffering
__global__ void gemm_f32_gfx906_double_buffer(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int M, const int N, const int K,
    const float alpha, const float beta
) {
    const int tid = threadIdx.x;
    const int block_row = blockIdx.y * DB_TILE_M;
    const int block_col = blockIdx.x * DB_TILE_N;
    
    // Double-buffered shared memory
    __shared__ float lds_tile_a[2][DB_TILE_K][DB_TILE_M + 1];
    __shared__ float lds_tile_b[2][DB_TILE_K][DB_TILE_N + 1];
    
    const int threads_per_row = DB_TILE_N / DB_THREAD_TILE_N;
    const int thread_row = (tid / threads_per_row) * DB_THREAD_TILE_M;
    const int thread_col = (tid % threads_per_row) * DB_THREAD_TILE_N;
    
    float acc[DB_THREAD_TILE_M][DB_THREAD_TILE_N] = {{0.0f}};
    
    // Load indices
    const int load_a_row = tid / (DB_TILE_K / 4);
    const int load_a_col = (tid % (DB_TILE_K / 4)) * 4;
    const int load_b_row = tid / (DB_TILE_N / 4);
    const int load_b_col = (tid % (DB_TILE_N / 4)) * 4;
    
    // Prefetch first tile
    if (load_a_row < DB_TILE_M && load_a_col < K) {
        float4 a_data = *reinterpret_cast<const float4*>(
            &A[(block_row + load_a_row) * K + load_a_col]);
        lds_tile_a[0][load_a_col][load_a_row] = a_data.x;
        lds_tile_a[0][load_a_col + 1][load_a_row] = a_data.y;
        lds_tile_a[0][load_a_col + 2][load_a_row] = a_data.z;
        lds_tile_a[0][load_a_col + 3][load_a_row] = a_data.w;
    }
    
    if (load_b_row < DB_TILE_K && load_b_col < DB_TILE_N) {
        float4 b_data = *reinterpret_cast<const float4*>(
            &B[load_b_row * N + block_col + load_b_col]);
        lds_tile_b[0][load_b_row][load_b_col] = b_data.x;
        lds_tile_b[0][load_b_row][load_b_col + 1] = b_data.y;
        lds_tile_b[0][load_b_row][load_b_col + 2] = b_data.z;
        lds_tile_b[0][load_b_row][load_b_col + 3] = b_data.w;
    }
    
    __syncthreads();
    
    // Main loop with double buffering
    int buffer_idx = 0;
    
    for (int k_tile = 0; k_tile < K; k_tile += DB_TILE_K) {
        int next_buffer = 1 - buffer_idx;
        int next_k_tile = k_tile + DB_TILE_K;
        
        // Load next tile
        if (next_k_tile < K) {
            if (load_a_row < DB_TILE_M && next_k_tile + load_a_col < K) {
                float4 a_data = *reinterpret_cast<const float4*>(
                    &A[(block_row + load_a_row) * K + next_k_tile + load_a_col]);
                lds_tile_a[next_buffer][load_a_col][load_a_row] = a_data.x;
                lds_tile_a[next_buffer][load_a_col + 1][load_a_row] = a_data.y;
                lds_tile_a[next_buffer][load_a_col + 2][load_a_row] = a_data.z;
                lds_tile_a[next_buffer][load_a_col + 3][load_a_row] = a_data.w;
            }
            
            if (load_b_row < DB_TILE_K && load_b_col < DB_TILE_N) {
                float4 b_data = *reinterpret_cast<const float4*>(
                    &B[(next_k_tile + load_b_row) * N + block_col + load_b_col]);
                lds_tile_b[next_buffer][load_b_row][load_b_col] = b_data.x;
                lds_tile_b[next_buffer][load_b_row][load_b_col + 1] = b_data.y;
                lds_tile_b[next_buffer][load_b_row][load_b_col + 2] = b_data.z;
                lds_tile_b[next_buffer][load_b_row][load_b_col + 3] = b_data.w;
            }
        }
        
        // Compute using current buffer
        #pragma unroll
        for (int k = 0; k < DB_TILE_K; k++) {
            float a_reg[DB_THREAD_TILE_M];
            float b_reg[DB_THREAD_TILE_N];
            
            #pragma unroll
            for (int m = 0; m < DB_THREAD_TILE_M; m++) {
                a_reg[m] = lds_tile_a[buffer_idx][k][thread_row + m];
            }
            
            #pragma unroll
            for (int n = 0; n < DB_THREAD_TILE_N; n++) {
                b_reg[n] = lds_tile_b[buffer_idx][k][thread_col + n];
            }
            
            #pragma unroll
            for (int m = 0; m < DB_THREAD_TILE_M; m++) {
                #pragma unroll
                for (int n = 0; n < DB_THREAD_TILE_N; n++) {
                    acc[m][n] = fmaf(a_reg[m], b_reg[n], acc[m][n]);
                }
            }
        }
        
        buffer_idx = next_buffer;
        __syncthreads();
    }
    
    // Store results
    #pragma unroll
    for (int m = 0; m < DB_THREAD_TILE_M; m++) {
        #pragma unroll
        for (int n = 0; n < DB_THREAD_TILE_N; n++) {
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