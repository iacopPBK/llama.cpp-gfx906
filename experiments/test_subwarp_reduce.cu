// Test warp_reduce_any<32> on 64-wide wavefront
// Critical test: does the reduction stay isolated to each 32-lane group?

#include <hip/hip_runtime.h>
#include <stdio.h>

// DPP implementations
__device__ int dpp_xor1(int value) {
    int result;
    asm volatile("s_nop 4\n v_mov_b32_dpp %0, %1 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf"
        : "=v"(result) : "v"(value) : "memory");
    return result;
}

__device__ int dpp_xor2(int value) {
    int result;
    asm volatile("s_nop 1\n v_mov_b32_dpp %0, %1 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf"
        : "=v"(result) : "v"(value) : "memory");
    return result;
}

__device__ int dpp_xor4(int value) {
    int result;
    asm volatile(
        "v_mov_b32 %0, %1\n s_nop 1\n"
        "v_mov_b32_dpp %0, %1 row_shl:4 row_mask:0xf bank_mask:0x5\n"
        "v_mov_b32_dpp %0, %1 row_shr:4 row_mask:0xf bank_mask:0xa\n"
        : "=v"(result) : "v"(value) : "memory");
    return result;
}

__device__ int dpp_xor8(int value) {
    int result;
    asm volatile("s_nop 1\n v_mov_b32_dpp %0, %1 row_ror:8 row_mask:0xf bank_mask:0xf"
        : "=v"(result) : "v"(value) : "memory");
    return result;
}

__device__ int dpp_xor16(int value) {
    int result;
    asm volatile(
        "ds_swizzle_b32 %0, %1 offset:swizzle(SWAP,16)\n s_waitcnt lgkmcnt(0)\n"
        : "=v"(result) : "v"(value) : "memory");
    return result;
}

// YOUR warp_reduce_any<32> implementation using DPP
__device__ int warp_reduce_any_32_dpp(int x) {
    x = dpp_xor16(x) || x;
    x = dpp_xor8(x) || x;
    x = dpp_xor4(x) || x;
    x = dpp_xor2(x) || x;
    x = dpp_xor1(x) || x;
    return x;
}

// Reference implementation using __shfl_xor with width=32
__device__ int warp_reduce_any_32_shfl(int x) {
    x = __shfl_xor(x, 16, 32) || x;
    x = __shfl_xor(x, 8, 32) || x;
    x = __shfl_xor(x, 4, 32) || x;
    x = __shfl_xor(x, 2, 32) || x;
    x = __shfl_xor(x, 1, 32) || x;
    return x;
}

__global__ void test_subwarp_reduce(int* results_dpp, int* results_shfl) {
    int lane = threadIdx.x;

    // Test case: ONLY lane 0 has a non-zero value in group 0
    //            ONLY lane 32 has a non-zero value in group 1
    // Expected: lanes 0-31 should all get 1, lanes 32-63 should all get 1
    //           (each group reduces its own non-zero lane)

    // Another test: Only lane 0 has value, lane 32 is zero
    // Expected: lanes 0-31 get 1, lanes 32-63 get 0
    // This tests if groups are truly isolated

    int input;
    if (lane == 0) {
        input = 1;  // Only lane 0 in group 0 has a value
    } else if (lane == 32) {
        input = 0;  // Lane 32 in group 1 has NO value (test isolation)
    } else {
        input = 0;
    }

    int result_dpp = warp_reduce_any_32_dpp(input);
    int result_shfl = warp_reduce_any_32_shfl(input);

    results_dpp[lane] = result_dpp;
    results_shfl[lane] = result_shfl;
}

int main() {
    int *d_dpp, *d_shfl;
    int h_dpp[64], h_shfl[64];

    hipMalloc(&d_dpp, 64 * sizeof(int));
    hipMalloc(&d_shfl, 64 * sizeof(int));

    test_subwarp_reduce<<<1, 64>>>(d_dpp, d_shfl);
    hipDeviceSynchronize();

    hipMemcpy(h_dpp, d_dpp, 64 * sizeof(int), hipMemcpyDeviceToHost);
    hipMemcpy(h_shfl, d_shfl, 64 * sizeof(int), hipMemcpyDeviceToHost);

    printf("Test: Only lane 0 has value=1, all others (including lane 32) have value=0\n");
    printf("Expected: Group 0 (lanes 0-31) should all return 1\n");
    printf("Expected: Group 1 (lanes 32-63) should all return 0 (isolated!)\n\n");

    printf("Results using DPP:\n");
    printf("Group 0 (lanes 0-31): ");
    int g0_dpp_ones = 0, g0_dpp_zeros = 0;
    for (int i = 0; i < 32; i++) {
        if (h_dpp[i] == 1) g0_dpp_ones++;
        else g0_dpp_zeros++;
    }
    printf("%d ones, %d zeros\n", g0_dpp_ones, g0_dpp_zeros);

    printf("Group 1 (lanes 32-63): ");
    int g1_dpp_ones = 0, g1_dpp_zeros = 0;
    for (int i = 32; i < 64; i++) {
        if (h_dpp[i] == 1) g1_dpp_ones++;
        else g1_dpp_zeros++;
    }
    printf("%d ones, %d zeros\n", g1_dpp_ones, g1_dpp_zeros);

    printf("\nResults using __shfl_xor (reference):\n");
    printf("Group 0 (lanes 0-31): ");
    int g0_shfl_ones = 0, g0_shfl_zeros = 0;
    for (int i = 0; i < 32; i++) {
        if (h_shfl[i] == 1) g0_shfl_ones++;
        else g0_shfl_zeros++;
    }
    printf("%d ones, %d zeros\n", g0_shfl_ones, g0_shfl_zeros);

    printf("Group 1 (lanes 32-63): ");
    int g1_shfl_ones = 0, g1_shfl_zeros = 0;
    for (int i = 32; i < 64; i++) {
        if (h_shfl[i] == 1) g1_shfl_ones++;
        else g1_shfl_zeros++;
    }
    printf("%d ones, %d zeros\n", g1_shfl_ones, g1_shfl_zeros);

    printf("\n=== VERDICT ===\n");
    bool dpp_ok = (g0_dpp_ones == 32 && g1_dpp_zeros == 32);
    bool shfl_ok = (g0_shfl_ones == 32 && g1_shfl_zeros == 32);

    if (dpp_ok) {
        printf("DPP implementation: CORRECT (groups isolated)\n");
    } else {
        printf("DPP implementation: BROKEN (group 1 got contaminated with %d ones)\n", g1_dpp_ones);
    }

    if (shfl_ok) {
        printf("__shfl_xor implementation: CORRECT (groups isolated)\n");
    } else {
        printf("__shfl_xor implementation: BROKEN (group 1 got contaminated with %d ones)\n", g1_shfl_ones);
    }

    hipFree(d_dpp);
    hipFree(d_shfl);
    return 0;
}
