// Trace DPP warp_reduce_any<32> step by step to find the bug

#include <hip/hip_runtime.h>
#include <stdio.h>

__device__ int dpp_xor16(int value) {
    int result;
    asm volatile(
        "ds_swizzle_b32 %0, %1 offset:swizzle(SWAP,16)\n s_waitcnt lgkmcnt(0)\n"
        : "=v"(result) : "v"(value) : "memory");
    return result;
}

__device__ int dpp_xor8(int value) {
    int result;
    asm volatile("s_nop 1\n v_mov_b32_dpp %0, %1 row_ror:8 row_mask:0xf bank_mask:0xf"
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

__device__ int dpp_xor2(int value) {
    int result;
    asm volatile("s_nop 1\n v_mov_b32_dpp %0, %1 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf"
        : "=v"(result) : "v"(value) : "memory");
    return result;
}

__device__ int dpp_xor1(int value) {
    int result;
    asm volatile("s_nop 4\n v_mov_b32_dpp %0, %1 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf"
        : "=v"(result) : "v"(value) : "memory");
    return result;
}

__global__ void trace_reduction(int* after_xor16, int* after_xor8, int* after_xor4,
                                 int* after_xor2, int* after_xor1) {
    int lane = threadIdx.x;

    // Input: only lane 0 has 1
    int x = (lane == 0) ? 1 : 0;

    // Step 1: XOR 16
    int xor16_val = dpp_xor16(x);
    x = xor16_val || x;
    after_xor16[lane] = x;

    // Step 2: XOR 8
    int xor8_val = dpp_xor8(x);
    x = xor8_val || x;
    after_xor8[lane] = x;

    // Step 3: XOR 4
    int xor4_val = dpp_xor4(x);
    x = xor4_val || x;
    after_xor4[lane] = x;

    // Step 4: XOR 2
    int xor2_val = dpp_xor2(x);
    x = xor2_val || x;
    after_xor2[lane] = x;

    // Step 5: XOR 1
    int xor1_val = dpp_xor1(x);
    x = xor1_val || x;
    after_xor1[lane] = x;
}

void print_lanes(const char* name, int* data) {
    printf("%s:\n", name);
    printf("  Lanes 0-15:  ");
    for (int i = 0; i < 16; i++) printf("%d", data[i]);
    printf("\n");
    printf("  Lanes 16-31: ");
    for (int i = 16; i < 32; i++) printf("%d", data[i]);
    printf("\n");
    printf("  Lanes 32-47: ");
    for (int i = 32; i < 48; i++) printf("%d", data[i]);
    printf("\n");
    printf("  Lanes 48-63: ");
    for (int i = 48; i < 64; i++) printf("%d", data[i]);
    printf("\n\n");
}

int main() {
    int *d_xor16, *d_xor8, *d_xor4, *d_xor2, *d_xor1;
    int h_xor16[64], h_xor8[64], h_xor4[64], h_xor2[64], h_xor1[64];

    hipMalloc(&d_xor16, 64 * sizeof(int));
    hipMalloc(&d_xor8, 64 * sizeof(int));
    hipMalloc(&d_xor4, 64 * sizeof(int));
    hipMalloc(&d_xor2, 64 * sizeof(int));
    hipMalloc(&d_xor1, 64 * sizeof(int));

    trace_reduction<<<1, 64>>>(d_xor16, d_xor8, d_xor4, d_xor2, d_xor1);
    hipDeviceSynchronize();

    hipMemcpy(h_xor16, d_xor16, 64 * sizeof(int), hipMemcpyDeviceToHost);
    hipMemcpy(h_xor8, d_xor8, 64 * sizeof(int), hipMemcpyDeviceToHost);
    hipMemcpy(h_xor4, d_xor4, 64 * sizeof(int), hipMemcpyDeviceToHost);
    hipMemcpy(h_xor2, d_xor2, 64 * sizeof(int), hipMemcpyDeviceToHost);
    hipMemcpy(h_xor1, d_xor1, 64 * sizeof(int), hipMemcpyDeviceToHost);

    printf("Tracing warp_reduce_any<32> step by step\n");
    printf("Input: lane 0 = 1, all others = 0\n");
    printf("Expected: after all steps, lanes 0-31 should all be 1\n\n");

    print_lanes("After XOR 16 (lane 0 ↔ lane 16)", h_xor16);
    print_lanes("After XOR 8 (within-row rotation)", h_xor8);
    print_lanes("After XOR 4 (bank shuffle)", h_xor4);
    print_lanes("After XOR 2 (quad permute)", h_xor2);
    print_lanes("After XOR 1 (quad permute)", h_xor1);

    int ones_in_group0 = 0;
    for (int i = 0; i < 32; i++) ones_in_group0 += h_xor1[i];
    printf("Final result: %d/32 lanes in group 0 have value 1\n", ones_in_group0);

    hipFree(d_xor16);
    hipFree(d_xor8);
    hipFree(d_xor4);
    hipFree(d_xor2);
    hipFree(d_xor1);
    return 0;
}
