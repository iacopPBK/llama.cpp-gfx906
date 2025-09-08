#pragma once

#include "common.cuh"
#include "gfx906-config.cuh"
#include "gfx906-wave-primitives.cuh"

#ifdef GGML_USE_HIP
#include <hip/hip_vector_types.h>
#endif

#ifdef GGML_HIP_GFX906_OPTIMIZED

// High-performance GEMM kernel for GFX906
// Targets 4-5 TFLOPS using 128x128x32 tiling with double buffering
// and full 64KB LDS utilization

#    define GEMM_GFX906_TILE_M            64
#    define GEMM_GFX906_TILE_N            64
#    define GEMM_GFX906_TILE_K            32
#    define GEMM_GFX906_THREADS_PER_BLOCK 256
#    define GEMM_GFX906_WAVES_PER_BLOCK   (GEMM_GFX906_THREADS_PER_BLOCK / GFX906_WAVE_SIZE)

// Double buffering configuration
#    define GEMM_GFX906_DOUBLE_BUFFER 1
#    define GEMM_GFX906_NUM_BUFFERS   2

// LDS allocation sizes (using full 64KB)
#    define GEMM_GFX906_LDS_A_SIZE  (GEMM_GFX906_TILE_M * GEMM_GFX906_TILE_K)
#    define GEMM_GFX906_LDS_B_SIZE  (GEMM_GFX906_TILE_K * GEMM_GFX906_TILE_N)
#    define GEMM_GFX906_LDS_PADDING 1  // Bank conflict avoidance (1 float = 4 bytes = 1 bank width)

// Thread tile configuration for register blocking
#    define GEMM_GFX906_THREAD_TILE_M 8
#    define GEMM_GFX906_THREAD_TILE_N 8

// Optimized GEMM kernel for FP32
template <int TILE_M = GEMM_GFX906_TILE_M, int TILE_N = GEMM_GFX906_TILE_N, int TILE_K = GEMM_GFX906_TILE_K>
__global__ void gemm_f32_gfx906(const float * __restrict__ A,
                                const float * __restrict__ B,
                                float * __restrict__ C,
                                const int   M,
                                const int   N,
                                const int   K,
                                const float alpha,
                                const float beta) {
    // Shared memory for double buffering
    extern __shared__ float smem_f32[];

    float * tile_a[GEMM_GFX906_NUM_BUFFERS];
    float * tile_b[GEMM_GFX906_NUM_BUFFERS];

    // Setup double buffer pointers
    const int lds_stride_a = TILE_M * (TILE_K + GEMM_GFX906_LDS_PADDING);
    const int lds_stride_b = TILE_K * (TILE_N + GEMM_GFX906_LDS_PADDING);

    tile_a[0] = smem_f32;
    tile_a[1] = smem_f32 + lds_stride_a;
    tile_b[0] = smem_f32 + 2 * lds_stride_a;
    tile_b[1] = smem_f32 + 2 * lds_stride_a + lds_stride_b;

    const int tid  = threadIdx.x;
    const int wid  = tid / GFX906_WAVE_SIZE;
    const int lane = tid % GFX906_WAVE_SIZE;

    // Block and thread tile indices
    const int block_row = blockIdx.y * TILE_M;
    const int block_col = blockIdx.x * TILE_N;

    // Thread-level tile position within block tile
    const int thread_row = (tid / (TILE_N / GEMM_GFX906_THREAD_TILE_N)) * GEMM_GFX906_THREAD_TILE_M;
    const int thread_col = (tid % (TILE_N / GEMM_GFX906_THREAD_TILE_N)) * GEMM_GFX906_THREAD_TILE_N;

    // Register blocking for accumulation
    float acc[GEMM_GFX906_THREAD_TILE_M][GEMM_GFX906_THREAD_TILE_N] = { 0.0f };

    // Prefetch first tiles
    int write_buf = 0;
    int read_buf  = 1;

// Load first tile of A using vectorized loads for coalescing
    // Each thread loads float4 (128 bits) at a time for perfect coalescing
    const int elements_per_thread = 4;  // float4
    const int total_elements = TILE_M * TILE_K;
    const int loads_per_block = (total_elements + elements_per_thread - 1) / elements_per_thread;
    
#    pragma unroll
    for (int i = tid; i < loads_per_block; i += GEMM_GFX906_THREADS_PER_BLOCK) {
        const int elem_idx = i * elements_per_thread;
        const int row = elem_idx / TILE_K;
        const int col = elem_idx % TILE_K;
        const int global_row = block_row + row;
        
        // Load float4 for better memory bandwidth
        float4 vals = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (global_row < M && col + 3 < K) {
            vals = *reinterpret_cast<const float4*>(&A[global_row * K + col]);
        } else {
            // Handle boundary with individual loads
            if (global_row < M) {
                if (col < K) vals.x = A[global_row * K + col];
                if (col + 1 < K) vals.y = A[global_row * K + col + 1];
                if (col + 2 < K) vals.z = A[global_row * K + col + 2];
                if (col + 3 < K) vals.w = A[global_row * K + col + 3];
            }
        }
        
        // Store to shared memory
        if (elem_idx < total_elements) {
            tile_a[write_buf][row * (TILE_K + GEMM_GFX906_LDS_PADDING) + col] = vals.x;
            if (elem_idx + 1 < total_elements && col + 1 < TILE_K)
                tile_a[write_buf][row * (TILE_K + GEMM_GFX906_LDS_PADDING) + col + 1] = vals.y;
            if (elem_idx + 2 < total_elements && col + 2 < TILE_K)
                tile_a[write_buf][row * (TILE_K + GEMM_GFX906_LDS_PADDING) + col + 2] = vals.z;
            if (elem_idx + 3 < total_elements && col + 3 < TILE_K)
                tile_a[write_buf][row * (TILE_K + GEMM_GFX906_LDS_PADDING) + col + 3] = vals.w;
        }
    }

// Load first tile of B using vectorized loads for coalescing
    const int b_total_elements = TILE_K * TILE_N;
    const int b_loads_per_block = (b_total_elements + elements_per_thread - 1) / elements_per_thread;
    
#    pragma unroll
    for (int i = tid; i < b_loads_per_block; i += GEMM_GFX906_THREADS_PER_BLOCK) {
        const int elem_idx = i * elements_per_thread;
        const int row = elem_idx / TILE_N;
        const int col = elem_idx % TILE_N;
        const int global_col = block_col + col;
        
        // Load float4 for better memory bandwidth
        float4 vals = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (row < K && global_col + 3 < N) {
            vals = *reinterpret_cast<const float4*>(&B[row * N + global_col]);
        } else {
            // Handle boundary with individual loads
            if (row < K) {
                if (global_col < N) vals.x = B[row * N + global_col];
                if (global_col + 1 < N) vals.y = B[row * N + global_col + 1];
                if (global_col + 2 < N) vals.z = B[row * N + global_col + 2];
                if (global_col + 3 < N) vals.w = B[row * N + global_col + 3];
            }
        }
        
        // Store to shared memory
        if (elem_idx < b_total_elements) {
            tile_b[write_buf][row * (TILE_N + GEMM_GFX906_LDS_PADDING) + col] = vals.x;
            if (elem_idx + 1 < b_total_elements && col + 1 < TILE_N)
                tile_b[write_buf][row * (TILE_N + GEMM_GFX906_LDS_PADDING) + col + 1] = vals.y;
            if (elem_idx + 2 < b_total_elements && col + 2 < TILE_N)
                tile_b[write_buf][row * (TILE_N + GEMM_GFX906_LDS_PADDING) + col + 2] = vals.z;
            if (elem_idx + 3 < b_total_elements && col + 3 < TILE_N)
                tile_b[write_buf][row * (TILE_N + GEMM_GFX906_LDS_PADDING) + col + 3] = vals.w;
        }
    }

    __syncthreads();

    // Main loop over K dimension with double buffering
    for (int k_tile = 0; k_tile < K; k_tile += TILE_K) {
        // Swap buffers
        write_buf = 1 - write_buf;
        read_buf  = 1 - read_buf;

        // Prefetch next tiles (if not last iteration)
        if (k_tile + TILE_K < K) {
// Async load next tile of A using vectorized loads
#    pragma unroll
            for (int i = tid; i < loads_per_block; i += GEMM_GFX906_THREADS_PER_BLOCK) {
                const int elem_idx = i * elements_per_thread;
                const int row = elem_idx / TILE_K;
                const int col = elem_idx % TILE_K;
                const int global_row = block_row + row;
                const int global_k = k_tile + TILE_K + col;
                
                float4 vals = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                if (global_row < M && global_k + 3 < K) {
                    vals = *reinterpret_cast<const float4*>(&A[global_row * K + global_k]);
                } else {
                    if (global_row < M) {
                        if (global_k < K) vals.x = A[global_row * K + global_k];
                        if (global_k + 1 < K) vals.y = A[global_row * K + global_k + 1];
                        if (global_k + 2 < K) vals.z = A[global_row * K + global_k + 2];
                        if (global_k + 3 < K) vals.w = A[global_row * K + global_k + 3];
                    }
                }
                
                if (elem_idx < total_elements) {
                    tile_a[write_buf][row * (TILE_K + GEMM_GFX906_LDS_PADDING) + col] = vals.x;
                    if (elem_idx + 1 < total_elements && col + 1 < TILE_K)
                        tile_a[write_buf][row * (TILE_K + GEMM_GFX906_LDS_PADDING) + col + 1] = vals.y;
                    if (elem_idx + 2 < total_elements && col + 2 < TILE_K)
                        tile_a[write_buf][row * (TILE_K + GEMM_GFX906_LDS_PADDING) + col + 2] = vals.z;
                    if (elem_idx + 3 < total_elements && col + 3 < TILE_K)
                        tile_a[write_buf][row * (TILE_K + GEMM_GFX906_LDS_PADDING) + col + 3] = vals.w;
                }
            }

// Async load next tile of B using vectorized loads
#    pragma unroll
            for (int i = tid; i < b_loads_per_block; i += GEMM_GFX906_THREADS_PER_BLOCK) {
                const int elem_idx = i * elements_per_thread;
                const int row = elem_idx / TILE_N;
                const int col = elem_idx % TILE_N;
                const int global_col = block_col + col;
                const int global_k = k_tile + TILE_K + row;
                
                float4 vals = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                if (global_k < K && global_col + 3 < N) {
                    vals = *reinterpret_cast<const float4*>(&B[global_k * N + global_col]);
                } else {
                    if (global_k < K) {
                        if (global_col < N) vals.x = B[global_k * N + global_col];
                        if (global_col + 1 < N) vals.y = B[global_k * N + global_col + 1];
                        if (global_col + 2 < N) vals.z = B[global_k * N + global_col + 2];
                        if (global_col + 3 < N) vals.w = B[global_k * N + global_col + 3];
                    }
                }
                
                if (elem_idx < b_total_elements) {
                    tile_b[write_buf][row * (TILE_N + GEMM_GFX906_LDS_PADDING) + col] = vals.x;
                    if (elem_idx + 1 < b_total_elements && col + 1 < TILE_N)
                        tile_b[write_buf][row * (TILE_N + GEMM_GFX906_LDS_PADDING) + col + 1] = vals.y;
                    if (elem_idx + 2 < b_total_elements && col + 2 < TILE_N)
                        tile_b[write_buf][row * (TILE_N + GEMM_GFX906_LDS_PADDING) + col + 2] = vals.z;
                    if (elem_idx + 3 < b_total_elements && col + 3 < TILE_N)
                        tile_b[write_buf][row * (TILE_N + GEMM_GFX906_LDS_PADDING) + col + 3] = vals.w;
                }
            }
        }

// Compute on current tiles
#    pragma unroll
        for (int k = 0; k < TILE_K; k++) {
            // Load A values for this thread's rows
            float a_reg[GEMM_GFX906_THREAD_TILE_M];
#    pragma unroll
            for (int m = 0; m < GEMM_GFX906_THREAD_TILE_M; m++) {
                a_reg[m] = tile_a[read_buf][(thread_row + m) * (TILE_K + GEMM_GFX906_LDS_PADDING) + k];
            }

            // Load B values for this thread's columns
            float b_reg[GEMM_GFX906_THREAD_TILE_N];
#    pragma unroll
            for (int n = 0; n < GEMM_GFX906_THREAD_TILE_N; n++) {
                b_reg[n] = tile_b[read_buf][k * (TILE_N + GEMM_GFX906_LDS_PADDING) + thread_col + n];
            }

// Perform outer product
#    pragma unroll
            for (int m = 0; m < GEMM_GFX906_THREAD_TILE_M; m++) {
#    pragma unroll
                for (int n = 0; n < GEMM_GFX906_THREAD_TILE_N; n++) {
                    acc[m][n] = __builtin_fmaf(a_reg[m], b_reg[n], acc[m][n]);
                }
            }
        }

        __syncthreads();
    }

    // Write results back to global memory
    const int out_row_base = block_row + thread_row;
    const int out_col_base = block_col + thread_col;

#    pragma unroll
    for (int m = 0; m < GEMM_GFX906_THREAD_TILE_M; m++) {
        const int out_row = out_row_base + m;
        if (out_row < M) {
#    pragma unroll
            for (int n = 0; n < GEMM_GFX906_THREAD_TILE_N; n++) {
                const int out_col = out_col_base + n;
                if (out_col < N) {
                    const int idx = out_row * N + out_col;
                    if (beta == 0.0f) {
                        C[idx] = alpha * acc[m][n];
                    } else {
                        C[idx] = alpha * acc[m][n] + beta * C[idx];
                    }
                }
            }
        }
    }
}

// Optimized GEMM kernel for FP16
template <int TILE_M = GEMM_GFX906_TILE_M, int TILE_N = GEMM_GFX906_TILE_N, int TILE_K = GEMM_GFX906_TILE_K>
__global__ void gemm_f16_gfx906(const half * __restrict__ A,
                                const half * __restrict__ B,
                                half * __restrict__ C,
                                const int   M,
                                const int   N,
                                const int   K,
                                const float alpha,
                                const float beta) {
    // Shared memory for double buffering
    extern __shared__ half smem_f16[];

    half * tile_a[GEMM_GFX906_NUM_BUFFERS];
    half * tile_b[GEMM_GFX906_NUM_BUFFERS];

    // Setup double buffer pointers
    const int lds_stride_a = TILE_M * (TILE_K + GEMM_GFX906_LDS_PADDING);
    const int lds_stride_b = TILE_K * (TILE_N + GEMM_GFX906_LDS_PADDING);

    tile_a[0] = smem_f16;
    tile_a[1] = smem_f16 + lds_stride_a;
    tile_b[0] = smem_f16 + 2 * lds_stride_a;
    tile_b[1] = smem_f16 + 2 * lds_stride_a + lds_stride_b;

    const int tid  = threadIdx.x;
    const int wid  = tid / GFX906_WAVE_SIZE;
    const int lane = tid % GFX906_WAVE_SIZE;

    // Block and thread tile indices
    const int block_row = blockIdx.y * TILE_M;
    const int block_col = blockIdx.x * TILE_N;

    // Thread-level tile position within block tile
    const int thread_row = (tid / (TILE_N / GEMM_GFX906_THREAD_TILE_N)) * GEMM_GFX906_THREAD_TILE_M;
    const int thread_col = (tid % (TILE_N / GEMM_GFX906_THREAD_TILE_N)) * GEMM_GFX906_THREAD_TILE_N;

    // Register blocking for accumulation - use float for better precision
    float acc[GEMM_GFX906_THREAD_TILE_M][GEMM_GFX906_THREAD_TILE_N] = { 0.0f };

    // Prefetch first tiles
    int write_buf = 0;
    int read_buf  = 1;

// Load first tile of A using vectorized loads
#    pragma unroll
    for (int i = tid; i < (TILE_M * TILE_K) / 2; i += GEMM_GFX906_THREADS_PER_BLOCK) {
        const int elem_idx   = i * 2;
        const int row        = elem_idx / TILE_K;
        const int col        = elem_idx % TILE_K;
        const int global_row = block_row + row;

        if (global_row < M && col + 1 < K) {
            *(half2 *) &tile_a[write_buf][row * (TILE_K + GEMM_GFX906_LDS_PADDING) + col] =
                *(half2 *) &A[global_row * K + col];
        } else {
            tile_a[write_buf][row * (TILE_K + GEMM_GFX906_LDS_PADDING) + col] = __float2half(0.0f);
            if (col + 1 < TILE_K) {
                tile_a[write_buf][row * (TILE_K + GEMM_GFX906_LDS_PADDING) + col + 1] = __float2half(0.0f);
            }
        }
    }

// Load first tile of B using vectorized loads
#    pragma unroll
    for (int i = tid; i < (TILE_K * TILE_N) / 2; i += GEMM_GFX906_THREADS_PER_BLOCK) {
        const int elem_idx   = i * 2;
        const int row        = elem_idx / TILE_N;
        const int col        = elem_idx % TILE_N;
        const int global_col = block_col + col;

        if (row < K && global_col + 1 < N) {
            *(half2 *) &tile_b[write_buf][row * (TILE_N + GEMM_GFX906_LDS_PADDING) + col] =
                *(half2 *) &B[row * N + global_col];
        } else {
            tile_b[write_buf][row * (TILE_N + GEMM_GFX906_LDS_PADDING) + col] = __float2half(0.0f);
            if (col + 1 < TILE_N) {
                tile_b[write_buf][row * (TILE_N + GEMM_GFX906_LDS_PADDING) + col + 1] = __float2half(0.0f);
            }
        }
    }

    __syncthreads();

    // Main loop over K dimension with double buffering
    for (int k_tile = 0; k_tile < K; k_tile += TILE_K) {
        // Swap buffers
        write_buf = 1 - write_buf;
        read_buf  = 1 - read_buf;

        // Prefetch next tiles (if not last iteration)
        if (k_tile + TILE_K < K) {
// Async load next tile of A
#    pragma unroll
            for (int i = tid; i < (TILE_M * TILE_K) / 2; i += GEMM_GFX906_THREADS_PER_BLOCK) {
                const int elem_idx   = i * 2;
                const int row        = elem_idx / TILE_K;
                const int col        = elem_idx % TILE_K;
                const int global_row = block_row + row;
                const int global_k   = k_tile + TILE_K + col;

                if (global_row < M && global_k + 1 < K) {
                    *(half2 *) &tile_a[write_buf][row * (TILE_K + GEMM_GFX906_LDS_PADDING) + col] =
                        *(half2 *) &A[global_row * K + global_k];
                }
            }

// Async load next tile of B
#    pragma unroll
            for (int i = tid; i < (TILE_K * TILE_N) / 2; i += GEMM_GFX906_THREADS_PER_BLOCK) {
                const int elem_idx   = i * 2;
                const int row        = elem_idx / TILE_N;
                const int col        = elem_idx % TILE_N;
                const int global_col = block_col + col;
                const int global_k   = k_tile + TILE_K + row;

                if (global_k < K && global_col + 1 < N) {
                    *(half2 *) &tile_b[write_buf][row * (TILE_N + GEMM_GFX906_LDS_PADDING) + col] =
                        *(half2 *) &B[global_k * N + global_col];
                }
            }
        }

// Compute on current tiles using V_DOT2_F32_F16 instruction
#    pragma unroll
        for (int k = 0; k < TILE_K; k += 2) {
            // Load A values for this thread's rows (2 FP16 values at a time)
            half2 a_reg[GEMM_GFX906_THREAD_TILE_M];
#    pragma unroll
            for (int m = 0; m < GEMM_GFX906_THREAD_TILE_M; m++) {
                a_reg[m] = *(half2 *) &tile_a[read_buf][(thread_row + m) * (TILE_K + GEMM_GFX906_LDS_PADDING) + k];
            }

            // Load B values for this thread's columns (2 consecutive K values packed)
            half2 b_reg[GEMM_GFX906_THREAD_TILE_N];
#    pragma unroll
            for (int n = 0; n < GEMM_GFX906_THREAD_TILE_N; n++) {
                // Load two consecutive K values (k and k+1) for each N position
                __half k0_val = tile_b[read_buf][k * (TILE_N + GEMM_GFX906_LDS_PADDING) + thread_col + n];
                __half k1_val = tile_b[read_buf][(k + 1) * (TILE_N + GEMM_GFX906_LDS_PADDING) + thread_col + n];
                b_reg[n] = make_half2(k0_val, k1_val);
            }

// Perform outer product using V_DOT2_F32_F16
#    pragma unroll
            for (int m = 0; m < GEMM_GFX906_THREAD_TILE_M; m++) {
#    pragma unroll
                for (int n = 0; n < GEMM_GFX906_THREAD_TILE_N; n++) {
                    acc[m][n] = gfx906_dot2_f16(*(uint32_t *) &a_reg[m], *(uint32_t *) &b_reg[n], acc[m][n]);
                }
            }
        }

        __syncthreads();
    }

    // Write results back to global memory
    const int out_row_base = block_row + thread_row;
    const int out_col_base = block_col + thread_col;

#    pragma unroll
    for (int m = 0; m < GEMM_GFX906_THREAD_TILE_M; m++) {
        const int out_row = out_row_base + m;
        if (out_row < M) {
#    pragma unroll
            for (int n = 0; n < GEMM_GFX906_THREAD_TILE_N; n++) {
                const int out_col = out_col_base + n;
                if (out_col < N) {
                    const int idx = out_row * N + out_col;
                    if (beta == 0.0f) {
                        C[idx] = __float2half(alpha * acc[m][n]);
                    } else {
                        C[idx] = __float2half(alpha * acc[m][n] + beta * __half2float(C[idx]));
                    }
                }
            }
        }
    }
}

// Kernel launcher for FP32 GEMM
inline void launch_gemm_f32_gfx906(const float * A,
                                   const float * B,
                                   float *       C,
                                   const int     M,
                                   const int     N,
                                   const int     K,
                                   const float   alpha,
                                   const float   beta,
                                   cudaStream_t  stream) {
    dim3 grid((N + GEMM_GFX906_TILE_N - 1) / GEMM_GFX906_TILE_N, (M + GEMM_GFX906_TILE_M - 1) / GEMM_GFX906_TILE_M);
    dim3 block(GEMM_GFX906_THREADS_PER_BLOCK);

    // Calculate shared memory size for double buffering
    // We need 2 buffers each for A and B tiles
    const size_t smem_size = GEMM_GFX906_NUM_BUFFERS *
                             (GEMM_GFX906_TILE_M * (GEMM_GFX906_TILE_K + GEMM_GFX906_LDS_PADDING) +
                              GEMM_GFX906_TILE_K * (GEMM_GFX906_TILE_N + GEMM_GFX906_LDS_PADDING)) *
                             sizeof(float);

    // Ensure we don't exceed LDS size
    if (smem_size > GFX906_LDS_SIZE) {
        // Fall back to smaller tile size or error
        printf("GEMM kernel requires %zu bytes of LDS, but only %d available\n", smem_size, GFX906_LDS_SIZE);
        return;
    }

    // Debug: print launch configuration
    static int launch_count = 0;
    if (launch_count++ < 5) {
        printf("Launching GEMM: grid(%d,%d) block(%d) smem=%zu bytes, M=%d N=%d K=%d\n", 
               grid.x, grid.y, block.x, smem_size, M, N, K);
    }

    gemm_f32_gfx906<<<grid, block, smem_size, stream>>>(A, B, C, M, N, K, alpha, beta);
    
    // Check for launch errors
    hipError_t err = hipGetLastError();
    if (err != hipSuccess && launch_count < 10) {
        printf("GEMM kernel launch error: %s\n", hipGetErrorString(err));
    }
}

// Kernel launcher for FP16 GEMM
inline void launch_gemm_f16_gfx906(const half * A,
                                   const half * B,
                                   half *       C,
                                   const int    M,
                                   const int    N,
                                   const int    K,
                                   const float  alpha,
                                   const float  beta,
                                   cudaStream_t stream) {
    dim3 grid((N + GEMM_GFX906_TILE_N - 1) / GEMM_GFX906_TILE_N, (M + GEMM_GFX906_TILE_M - 1) / GEMM_GFX906_TILE_M);
    dim3 block(GEMM_GFX906_THREADS_PER_BLOCK);

    // Calculate shared memory size for double buffering
    // We need 2 buffers each for A and B tiles
    const size_t smem_size = GEMM_GFX906_NUM_BUFFERS *
                             (GEMM_GFX906_TILE_M * (GEMM_GFX906_TILE_K + GEMM_GFX906_LDS_PADDING) +
                              GEMM_GFX906_TILE_K * (GEMM_GFX906_TILE_N + GEMM_GFX906_LDS_PADDING)) *
                             sizeof(half);

    // Ensure we don't exceed LDS size
    if (smem_size > GFX906_LDS_SIZE) {
        // Fall back to smaller tile size or error
        printf("GEMM kernel requires %zu bytes of LDS, but only %d available\n", smem_size, GFX906_LDS_SIZE);
        return;
    }

    gemm_f16_gfx906<<<grid, block, smem_size, stream>>>(A, B, C, M, N, K, alpha, beta);
}

#endif  // GGML_HIP_GFX906_OPTIMIZED
