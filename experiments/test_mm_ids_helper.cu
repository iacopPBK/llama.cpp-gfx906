// Test to replicate mm_ids_helper behavior with n_expert_used=32 on warp_size=64
// This tests the optimized path logic

#include <hip/hip_runtime.h>
#include <stdio.h>
#include <climits>

// Replicate the DPP-based warp_reduce_any<32>
__device__ int dpp_xor1(int value) {
    int result;
    asm volatile(
        "s_nop 4\n"
        "v_mov_b32_dpp %0, %1 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf"
        : "=v"(result) : "v"(value) : "memory"
    );
    return result;
}

__device__ int dpp_xor2(int value) {
    int result;
    asm volatile(
        "s_nop 1\n"
        "v_mov_b32_dpp %0, %1 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf"
        : "=v"(result) : "v"(value) : "memory"
    );
    return result;
}

__device__ int dpp_xor4(int value) {
    int result;
    asm volatile(
        "v_mov_b32 %0, %1\n"
        "s_nop 1\n"
        "v_mov_b32_dpp %0, %1 row_shl:4 row_mask:0xf bank_mask:0x5\n"
        "v_mov_b32_dpp %0, %1 row_shr:4 row_mask:0xf bank_mask:0xa\n"
        : "=v"(result) : "v"(value) : "memory"
    );
    return result;
}

__device__ int dpp_xor8(int value) {
    int result;
    asm volatile(
        "s_nop 1\n"
        "v_mov_b32_dpp %0, %1 row_ror:8 row_mask:0xf bank_mask:0xf"
        : "=v"(result) : "v"(value) : "memory"
    );
    return result;
}

__device__ int dpp_xor16(int value) {
    int result;
    asm volatile(
        "ds_swizzle_b32 %0, %1 offset:swizzle(SWAP,16)\n"
        "s_waitcnt lgkmcnt(0)\n"
        : "=v"(result) : "v"(value) : "memory"
    );
    return result;
}

// warp_reduce_any<32> using DPP
__device__ int warp_reduce_any_32(int x) {
    // width=32, so offsets are 16, 8, 4, 2, 1
    x = dpp_xor16(x) || x;  // offset=16
    x = dpp_xor8(x) || x;   // offset=8
    x = dpp_xor4(x) || x;   // offset=4
    x = dpp_xor2(x) || x;   // offset=2
    x = dpp_xor1(x) || x;   // offset=1
    return x;
}

// warp_reduce_any<64> using DPP (full warp)
__device__ int warp_reduce_any_64(int x) {
    x = __shfl_xor(x, 32, 64) || x;  // offset=32
    x = dpp_xor16(x) || x;           // offset=16
    x = dpp_xor8(x) || x;            // offset=8
    x = dpp_xor4(x) || x;            // offset=4
    x = dpp_xor2(x) || x;            // offset=2
    x = dpp_xor1(x) || x;            // offset=1
    return x;
}

// Simplified mm_ids_helper optimized path for n_expert_used=32
__global__ void test_mm_ids_helper_optimized(
    const int32_t* ids,      // Input: routing decisions [n_tokens, n_expert_used]
    int32_t* ids_src1,       // Output: remapped indices
    int32_t* debug_output,   // Debug: intermediate values
    int n_tokens,
    int si1                  // stride for ids
) {
    constexpr int warp_size = 64;
    constexpr int n_expert_used = 32;
    constexpr int neu_padded = 32;

    int expert = blockIdx.x;  // Which expert this block handles

    // Shared memory for storing results
    extern __shared__ int store[];

    int nex_prev = 0;
    int it_compact = 0;

    // The optimized loop: process warp_size/neu_padded = 2 tokens per iteration
    for (int it0 = 0; it0 < n_tokens; it0 += warp_size / neu_padded) {
        int it = it0 + threadIdx.x / neu_padded;  // Token index (0 or 1 for first iteration)
        int iex = threadIdx.x % neu_padded;       // Expert slot index (0-31)

        // Read which expert is used at this position
        int expert_used = (it < n_tokens) ? ids[it * si1 + iex] : INT_MAX;
        int iex_used = (expert_used == expert) ? iex : -1;

        // Count experts with lower indices
        nex_prev += (expert_used < expert) ? 1 : 0;

        // Debug: store intermediate values
        if (expert == 0) {
            debug_output[threadIdx.x * 4 + 0] = it;
            debug_output[threadIdx.x * 4 + 1] = iex;
            debug_output[threadIdx.x * 4 + 2] = expert_used;
            debug_output[threadIdx.x * 4 + 3] = iex_used;
        }

        // THIS IS THE CRITICAL PART: sub-warp reduction
        int it_compact_add_self = warp_reduce_any_32(iex_used != -1);

        // Prefix sum across token groups
        int it_compact_add_lower = 0;
        for (int offset = neu_padded; offset < warp_size; offset += neu_padded) {
            int tmp = __shfl_up_sync(0xFFFFFFFFFFFFFFFFULL, it_compact_add_self, offset, warp_size);
            if (threadIdx.x >= offset) {
                it_compact_add_lower += tmp;
            }
        }

        if (iex_used != -1) {
            store[it_compact + it_compact_add_lower] = it * 1000 + iex_used;  // Encode for debugging
        }

        it_compact += __shfl_sync(0xFFFFFFFFFFFFFFFFULL, it_compact_add_lower + it_compact_add_self, warp_size - 1, warp_size);
    }

    // Write output
    nex_prev = 0;  // Simplified: just use warp_reduce for sum
    for (int i = 0; i < 64; i++) {
        nex_prev += (threadIdx.x == i) ? nex_prev : 0;
    }

    for (int itc = threadIdx.x; itc < it_compact; itc += warp_size) {
        int stored = store[itc];
        int it = stored / 1000;
        int iex_used = stored % 1000;
        ids_src1[itc] = it * 1 + iex_used;  // Simplified output
    }
}

int main() {
    // Test case: 2 tokens, 32 experts used per token
    // ids[token][slot] = which expert is routed to that slot

    int n_tokens = 2;
    int n_expert_used = 32;
    int n_experts = 32;

    // Create input: each token routes to all 32 experts (slots 0-31 map to experts 0-31)
    int h_ids[64];  // 2 tokens * 32 slots
    for (int t = 0; t < n_tokens; t++) {
        for (int e = 0; e < n_expert_used; e++) {
            h_ids[t * n_expert_used + e] = e;  // Token t, slot e -> expert e
        }
    }

    printf("Input ids (routing decisions):\n");
    printf("Token 0: slots 0-31 map to experts 0-31\n");
    printf("Token 1: slots 0-31 map to experts 0-31\n\n");

    int *d_ids, *d_ids_src1, *d_debug;
    hipMalloc(&d_ids, 64 * sizeof(int));
    hipMalloc(&d_ids_src1, 64 * sizeof(int));
    hipMalloc(&d_debug, 64 * 4 * sizeof(int));

    hipMemcpy(d_ids, h_ids, 64 * sizeof(int), hipMemcpyHostToDevice);
    hipMemset(d_ids_src1, 0xFF, 64 * sizeof(int));  // Initialize with -1

    // Launch for expert 0 only (simplification)
    size_t shared_size = n_tokens * sizeof(int);
    test_mm_ids_helper_optimized<<<1, 64, shared_size>>>(
        d_ids, d_ids_src1, d_debug, n_tokens, n_expert_used
    );
    hipDeviceSynchronize();

    // Check for errors
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        printf("Kernel error: %s\n", hipGetErrorString(err));
        return 1;
    }

    int h_ids_src1[64];
    int h_debug[64 * 4];
    hipMemcpy(h_ids_src1, d_ids_src1, 64 * sizeof(int), hipMemcpyDeviceToHost);
    hipMemcpy(h_debug, d_debug, 64 * 4 * sizeof(int), hipMemcpyDeviceToHost);

    printf("Debug output (for expert 0):\n");
    printf("Lane | it | iex | expert_used | iex_used\n");
    printf("-----|----|----|-------------|----------\n");
    for (int i = 0; i < 64; i += 8) {
        printf("%4d | %2d | %2d | %11d | %8d\n",
               i, h_debug[i*4+0], h_debug[i*4+1], h_debug[i*4+2], h_debug[i*4+3]);
    }

    printf("\nOutput ids_src1 (first 10 values):\n");
    for (int i = 0; i < 10; i++) {
        printf("ids_src1[%d] = %d\n", i, h_ids_src1[i]);
    }

    hipFree(d_ids);
    hipFree(d_ids_src1);
    hipFree(d_debug);

    return 0;
}
