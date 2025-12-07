// Test: MXFP4 vectorized load strategies
// Goal: Find fastest way to load misaligned MXFP4 data
//
// MXFP4 block: { uint8_t e; uint8_t qs[16]; } = 17 bytes
// Block alignment cycles: 0, 17, 34, 51, 68... mod 4 = 0, 1, 2, 3, 0...
// qs offset from block: 1 byte
// So qs alignment: 1, 2, 3, 0, 1, 2, 3, 0... (every 4th block aligned!)

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstring>

#define BLOCK_SIZE 256
#define NUM_BLOCKS 4096
#define ITERS 10000

// Simulate MXFP4 block structure
struct block_mxfp4 {
    uint8_t e;
    uint8_t qs[16];
};

// ============================================================================
// V0: Current implementation (4 separate byte loads)
// ============================================================================
__device__ __forceinline__ int load_v0_bytes(const uint8_t* qs, int idx) {
    int x32  = qs[4*idx + 0] <<  0;
    x32     |= qs[4*idx + 1] <<  8;
    x32     |= qs[4*idx + 2] << 16;
    x32     |= qs[4*idx + 3] << 24;
    return x32;
}

// ============================================================================
// V1: Unaligned 32-bit load (let hardware handle it)
// ============================================================================
__device__ __forceinline__ int load_v1_unaligned(const uint8_t* qs, int idx) {
    // Force unaligned load - GPU should handle this
    return *((const int*)(qs + 4*idx));
}

// ============================================================================
// V2: Use __builtin for unaligned load
// ============================================================================
__device__ __forceinline__ int load_v2_builtin(const uint8_t* qs, int idx) {
    int result;
    memcpy(&result, qs + 4*idx, 4);
    return result;
}

// ============================================================================
// V3: Conditional aligned load (check alignment at runtime)
// ============================================================================
__device__ __forceinline__ int load_v3_conditional(const uint8_t* qs, int idx) {
    const uint8_t* addr = qs + 4*idx;
    if (((uintptr_t)addr & 3) == 0) {
        // Aligned - direct load
        return *((const int*)addr);
    } else {
        // Misaligned - byte loads
        int x32  = addr[0] <<  0;
        x32     |= addr[1] <<  8;
        x32     |= addr[2] << 16;
        x32     |= addr[3] << 24;
        return x32;
    }
}

// ============================================================================
// V4: Vectorized int2 load when 8-byte aligned, else bytes
// ============================================================================
__device__ __forceinline__ int2 load_v4_int2(const uint8_t* qs, int idx) {
    const uint8_t* addr = qs + 8*idx;
    if (((uintptr_t)addr & 7) == 0) {
        // 8-byte aligned - single int2 load
        return *((const int2*)addr);
    } else {
        // Fallback to byte loads
        int2 result;
        result.x  = addr[0] <<  0;
        result.x |= addr[1] <<  8;
        result.x |= addr[2] << 16;
        result.x |= addr[3] << 24;
        result.y  = addr[4] <<  0;
        result.y |= addr[5] <<  8;
        result.y |= addr[6] << 16;
        result.y |= addr[7] << 24;
        return result;
    }
}

// ============================================================================
// V5: Branchless - always use ds_read if in LDS, global_load otherwise
// Use inline asm to force specific load width
// ============================================================================
__device__ __forceinline__ int load_v5_asm_dword(const uint8_t* qs, int idx) {
    const uint8_t* addr = qs + 4*idx;
    int result;
    // Force a single dword load regardless of alignment
    // The hardware will handle misalignment
    asm volatile(
        "global_load_dword %0, %1, off\n"
        "s_waitcnt vmcnt(0)\n"
        : "=v"(result)
        : "v"(addr)
        : "memory"
    );
    return result;
}

// ============================================================================
// V6: Load int4 (16 bytes) when 16-byte aligned
// ============================================================================
__device__ __forceinline__ int4 load_v6_int4(const uint8_t* qs) {
    if (((uintptr_t)qs & 15) == 0) {
        // 16-byte aligned - single int4 load
        return *((const int4*)qs);
    } else {
        // Fallback
        int4 result;
        result.x = load_v0_bytes(qs, 0);
        result.y = load_v0_bytes(qs, 1);
        result.z = load_v0_bytes(qs, 2);
        result.w = load_v0_bytes(qs, 3);
        return result;
    }
}

// ============================================================================
// V7: Precompute block alignment and use lookup
// Every 4th block has qs 4-byte aligned
// ============================================================================
__device__ __forceinline__ int load_v7_precomputed(const block_mxfp4* blocks, int block_idx, int qs_idx) {
    // block_idx % 4 determines alignment of qs:
    // 0 -> qs offset 1 (misaligned)
    // 1 -> qs offset 18 mod 4 = 2 (misaligned)
    // 2 -> qs offset 35 mod 4 = 3 (misaligned)
    // 3 -> qs offset 52 mod 4 = 0 (ALIGNED!)

    const uint8_t* qs = blocks[block_idx].qs;

    if ((block_idx & 3) == 3) {
        // This block's qs is 4-byte aligned
        return ((const int*)qs)[qs_idx];
    } else {
        return load_v0_bytes(qs, qs_idx);
    }
}

// ============================================================================
// KERNELS
// ============================================================================

__global__ void kernel_v0_bytes(const block_mxfp4* blocks, int* output, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int sum = 0;
    for (int iter = 0; iter < ITERS; iter++) {
        int block_idx = (tid + iter) % n;
        const uint8_t* qs = blocks[block_idx].qs;
        for (int i = 0; i < 4; i++) {
            sum += load_v0_bytes(qs, i);
        }
    }
    output[tid] = sum;
}

__global__ void kernel_v1_unaligned(const block_mxfp4* blocks, int* output, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int sum = 0;
    for (int iter = 0; iter < ITERS; iter++) {
        int block_idx = (tid + iter) % n;
        const uint8_t* qs = blocks[block_idx].qs;
        for (int i = 0; i < 4; i++) {
            sum += load_v1_unaligned(qs, i);
        }
    }
    output[tid] = sum;
}

__global__ void kernel_v2_builtin(const block_mxfp4* blocks, int* output, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int sum = 0;
    for (int iter = 0; iter < ITERS; iter++) {
        int block_idx = (tid + iter) % n;
        const uint8_t* qs = blocks[block_idx].qs;
        for (int i = 0; i < 4; i++) {
            sum += load_v2_builtin(qs, i);
        }
    }
    output[tid] = sum;
}

__global__ void kernel_v3_conditional(const block_mxfp4* blocks, int* output, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int sum = 0;
    for (int iter = 0; iter < ITERS; iter++) {
        int block_idx = (tid + iter) % n;
        const uint8_t* qs = blocks[block_idx].qs;
        for (int i = 0; i < 4; i++) {
            sum += load_v3_conditional(qs, i);
        }
    }
    output[tid] = sum;
}

__global__ void kernel_v4_int2(const block_mxfp4* blocks, int* output, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int sum = 0;
    for (int iter = 0; iter < ITERS; iter++) {
        int block_idx = (tid + iter) % n;
        const uint8_t* qs = blocks[block_idx].qs;
        int2 v0 = load_v4_int2(qs, 0);
        int2 v1 = load_v4_int2(qs, 1);
        sum += v0.x + v0.y + v1.x + v1.y;
    }
    output[tid] = sum;
}

__global__ void kernel_v5_asm(const block_mxfp4* blocks, int* output, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int sum = 0;
    for (int iter = 0; iter < ITERS; iter++) {
        int block_idx = (tid + iter) % n;
        const uint8_t* qs = blocks[block_idx].qs;
        for (int i = 0; i < 4; i++) {
            sum += load_v5_asm_dword(qs, i);
        }
    }
    output[tid] = sum;
}

__global__ void kernel_v7_precomputed(const block_mxfp4* blocks, int* output, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int sum = 0;
    for (int iter = 0; iter < ITERS; iter++) {
        int block_idx = (tid + iter) % n;
        for (int i = 0; i < 4; i++) {
            sum += load_v7_precomputed(blocks, block_idx, i);
        }
    }
    output[tid] = sum;
}

// ============================================================================
// BENCHMARK
// ============================================================================

void bench(const char* name, void(*kernel)(const block_mxfp4*, int*, int),
           const block_mxfp4* d_blocks, int* d_output, int n) {
    // Warmup
    kernel<<<1, BLOCK_SIZE>>>(d_blocks, d_output, n);
    hipDeviceSynchronize();

    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);

    hipEventRecord(start);
    kernel<<<1, BLOCK_SIZE>>>(d_blocks, d_output, n);
    hipEventRecord(stop);
    hipEventSynchronize(stop);

    float ms;
    hipEventElapsedTime(&ms, start, stop);

    double loads_per_thread = ITERS * 4;  // 4 loads per iteration
    double total_loads = BLOCK_SIZE * loads_per_thread;
    double gops = total_loads / (ms * 1e6);

    printf("  %-25s: %7.3f ms  (%6.2f G loads/s)\n", name, ms, gops);

    hipEventDestroy(start);
    hipEventDestroy(stop);
}

int main() {
    printf("MXFP4 Load Strategy Benchmark (GFX906)\n");
    printf("======================================\n");
    printf("Block structure: { uint8_t e; uint8_t qs[16]; } = 17 bytes\n");
    printf("qs alignment cycles: 1,2,3,0,1,2,3,0... (every 4th block aligned)\n\n");

    // Allocate
    block_mxfp4* h_blocks = new block_mxfp4[NUM_BLOCKS];
    for (int i = 0; i < NUM_BLOCKS; i++) {
        h_blocks[i].e = i & 0xFF;
        for (int j = 0; j < 16; j++) {
            h_blocks[i].qs[j] = (i + j) & 0xFF;
        }
    }

    block_mxfp4* d_blocks;
    int* d_output;
    hipMalloc(&d_blocks, NUM_BLOCKS * sizeof(block_mxfp4));
    hipMalloc(&d_output, BLOCK_SIZE * sizeof(int));
    hipMemcpy(d_blocks, h_blocks, NUM_BLOCKS * sizeof(block_mxfp4), hipMemcpyHostToDevice);

    printf("Benchmarks (%d iters, %d threads, %d blocks):\n", ITERS, BLOCK_SIZE, NUM_BLOCKS);

    bench("V0: 4 byte loads", kernel_v0_bytes, d_blocks, d_output, NUM_BLOCKS);
    bench("V1: unaligned dword", kernel_v1_unaligned, d_blocks, d_output, NUM_BLOCKS);
    bench("V2: memcpy builtin", kernel_v2_builtin, d_blocks, d_output, NUM_BLOCKS);
    bench("V3: conditional align", kernel_v3_conditional, d_blocks, d_output, NUM_BLOCKS);
    bench("V4: int2 when aligned", kernel_v4_int2, d_blocks, d_output, NUM_BLOCKS);
    bench("V5: asm global_load", kernel_v5_asm, d_blocks, d_output, NUM_BLOCKS);
    bench("V7: precomputed align", kernel_v7_precomputed, d_blocks, d_output, NUM_BLOCKS);

    // Verify correctness
    int h_output[BLOCK_SIZE];
    hipMemcpy(h_output, d_output, BLOCK_SIZE * sizeof(int), hipMemcpyDeviceToHost);
    printf("\nFirst output value: %d (for correctness check)\n", h_output[0]);

    hipFree(d_blocks);
    hipFree(d_output);
    delete[] h_blocks;

    return 0;
}
