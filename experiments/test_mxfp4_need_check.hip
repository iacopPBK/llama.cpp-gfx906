// Test: Why does MXFP4 need_check optimization regress while Q8 improves?
// Focus: Isolate the LDS write conflict issue

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>

#define MMQ_Y 64
#define NWARPS 4
#define WARP_SIZE 64
#define THREADS_PER_ROW 32
#define NROWS (WARP_SIZE / THREADS_PER_ROW)  // = 2
#define ITERATIONS_PER_WARP (MMQ_Y / (NROWS * NWARPS))  // = 64 / (2*4) = 8

// Simulated tile size
#define TILE_STRIDE 68  // MMQ_MMA_TILE_X_K_Q8_1

// ============================================================================
// ORIGINAL MXFP4 pattern (causes LDS conflicts when need_check=true)
// ============================================================================
template<bool need_check>
__global__ void load_tiles_original(int* x_tile, int i_max, int num_iters) {
    for (int iter = 0; iter < num_iters; iter++) {
        const int txi = threadIdx.x % THREADS_PER_ROW;

        #pragma unroll
        for (int i0 = 0; i0 < MMQ_Y; i0 += NROWS * NWARPS) {
            int i = i0 + threadIdx.y * NROWS + threadIdx.x / THREADS_PER_ROW;

            if (need_check) {
                i = min(i, i_max);  // ALL OOB threads clamp to same i!
            }

            // Simulate loading and decompressing MXFP4 data
            int val = i * 1000 + txi;  // Some dummy value

            // Write to LDS - ALL OOB threads write to x_tile[i_max * TILE_STRIDE + ...]
            x_tile[i * TILE_STRIDE + txi] = val;
            x_tile[i * TILE_STRIDE + txi + 4] = val + 1;
        }
    }
}

// ============================================================================
// OPTIMIZED MXFP4 pattern (each thread writes to unique slot)
// ============================================================================
template<bool need_check>
__global__ void load_tiles_optimized(int* x_tile, int i_max, int num_iters) {
    for (int iter = 0; iter < num_iters; iter++) {
        const int txi = threadIdx.x % THREADS_PER_ROW;

        #pragma unroll
        for (int i0 = 0; i0 < MMQ_Y; i0 += NROWS * NWARPS) {
            // Separate slot index from read index
            const int i_slot = i0 + threadIdx.y * NROWS + threadIdx.x / THREADS_PER_ROW;
            const int i_read = need_check ? min(i_slot, i_max) : i_slot;
            const bool oob = need_check && (i_slot > i_max);

            // Simulate loading from i_read position
            int val = i_read * 1000 + txi;  // Read from valid position

            // Write to unique slot, zeros if OOB
            x_tile[i_slot * TILE_STRIDE + txi] = oob ? 0 : val;
            x_tile[i_slot * TILE_STRIDE + txi + 4] = oob ? 0 : (val + 1);
        }
    }
}

// ============================================================================
// Version with lookup table (simulates MXFP4 decode)
// ============================================================================
__device__ __forceinline__ int2 fake_mxfp4_decode(int val) {
    // Simulate the v_perm lookup (actually fast)
    int x = __byte_perm(val, val, 0x3210);
    int y = __byte_perm(val, val, 0x7654);
    return make_int2(x, y);
}

template<bool need_check>
__global__ void load_tiles_original_with_decode(int* x_tile, int i_max, int num_iters) {
    for (int iter = 0; iter < num_iters; iter++) {
        const int txi = threadIdx.x % THREADS_PER_ROW;

        #pragma unroll
        for (int i0 = 0; i0 < MMQ_Y; i0 += NROWS * NWARPS) {
            int i = i0 + threadIdx.y * NROWS + threadIdx.x / THREADS_PER_ROW;

            if (need_check) {
                i = min(i, i_max);
            }

            // Simulate MXFP4 decode
            int2 v = fake_mxfp4_decode(i * 1000 + txi);

            x_tile[i * TILE_STRIDE + txi] = v.x;
            x_tile[i * TILE_STRIDE + txi + 4] = v.y;
        }
    }
}

template<bool need_check>
__global__ void load_tiles_optimized_with_decode(int* x_tile, int i_max, int num_iters) {
    for (int iter = 0; iter < num_iters; iter++) {
        const int txi = threadIdx.x % THREADS_PER_ROW;

        #pragma unroll
        for (int i0 = 0; i0 < MMQ_Y; i0 += NROWS * NWARPS) {
            const int i_slot = i0 + threadIdx.y * NROWS + threadIdx.x / THREADS_PER_ROW;
            const int i_read = need_check ? min(i_slot, i_max) : i_slot;
            const bool oob = need_check && (i_slot > i_max);

            // Decode from i_read
            int2 v = fake_mxfp4_decode(i_read * 1000 + txi);

            // Write to i_slot
            x_tile[i_slot * TILE_STRIDE + txi] = oob ? 0 : v.x;
            x_tile[i_slot * TILE_STRIDE + txi + 4] = oob ? 0 : v.y;
        }
    }
}

// ============================================================================
// Test with REAL OOB pattern (simulate partial tile at end)
// ============================================================================
void benchmark(const char* name, void(*kernel)(int*, int, int), int i_max, int num_iters) {
    int* d_tile;
    hipMalloc(&d_tile, MMQ_Y * TILE_STRIDE * sizeof(int));

    dim3 block(WARP_SIZE, NWARPS);

    // Warmup
    kernel<<<1, block>>>(d_tile, i_max, 10);
    hipDeviceSynchronize();

    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);

    hipEventRecord(start);
    kernel<<<1, block>>>(d_tile, i_max, num_iters);
    hipEventRecord(stop);
    hipEventSynchronize(stop);

    float ms;
    hipEventElapsedTime(&ms, start, stop);

    printf("  %-40s: %.3f ms\n", name, ms);

    hipEventDestroy(start);
    hipEventDestroy(stop);
    hipFree(d_tile);
}

int main() {
    printf("MXFP4 need_check Optimization Analysis\n");
    printf("=======================================\n");
    printf("Block: %d threads/warp x %d warps = %d threads\n", WARP_SIZE, NWARPS, WARP_SIZE * NWARPS);
    printf("MMQ_Y: %d, NROWS: %d, iterations: %d\n\n", MMQ_Y, NROWS, ITERATIONS_PER_WARP);

    const int NUM_ITERS = 100000;

    // Test 1: Full tile (need_check=false, i_max doesn't matter)
    printf("Test 1: Full tile (need_check=false)\n");
    benchmark("Original (need_check=false)", load_tiles_original<false>, 63, NUM_ITERS);
    benchmark("Optimized (need_check=false)", load_tiles_optimized<false>, 63, NUM_ITERS);
    printf("\n");

    // Test 2: Partial tile, many OOB (need_check=true, i_max=15 means 48 threads OOB)
    printf("Test 2: Partial tile, 75%% OOB (i_max=15, need_check=true)\n");
    benchmark("Original (need_check=true)", load_tiles_original<true>, 15, NUM_ITERS);
    benchmark("Optimized (need_check=true)", load_tiles_optimized<true>, 15, NUM_ITERS);
    printf("\n");

    // Test 3: Partial tile, few OOB (i_max=60, only 3 threads OOB)
    printf("Test 3: Partial tile, 6%% OOB (i_max=60, need_check=true)\n");
    benchmark("Original (need_check=true, 6%% OOB)", load_tiles_original<true>, 60, NUM_ITERS);
    benchmark("Optimized (need_check=true, 6%% OOB)", load_tiles_optimized<true>, 60, NUM_ITERS);
    printf("\n");

    // Test 4: With decode (simulates MXFP4 lookup)
    printf("Test 4: With MXFP4 decode, 75%% OOB\n");
    benchmark("Original+decode (75%% OOB)", load_tiles_original_with_decode<true>, 15, NUM_ITERS);
    benchmark("Optimized+decode (75%% OOB)", load_tiles_optimized_with_decode<true>, 15, NUM_ITERS);
    printf("\n");

    printf("Test 5: With MXFP4 decode, 6%% OOB\n");
    benchmark("Original+decode (6%% OOB)", load_tiles_original_with_decode<true>, 60, NUM_ITERS);
    benchmark("Optimized+decode (6%% OOB)", load_tiles_optimized_with_decode<true>, 60, NUM_ITERS);
    printf("\n");

    printf("Analysis:\n");
    printf("---------\n");
    printf("If Original is SLOWER than Optimized when need_check=true,\n");
    printf("then the LDS conflict hypothesis is confirmed.\n");
    printf("\n");
    printf("The key difference from Q8:\n");
    printf("- Q8 writes 1 value per thread per iteration\n");
    printf("- MXFP4 writes 2 values (int2) per thread per iteration\n");
    printf("- This doubles the conflict pressure but also the benefit!\n");

    return 0;
}
