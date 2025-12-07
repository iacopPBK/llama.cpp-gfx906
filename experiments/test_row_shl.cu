// Test row_shl:4 behavior in isolation

#include <hip/hip_runtime.h>
#include <stdio.h>

__global__ void test_row_shl(int* input_out, int* shl_out) {
    int lane = threadIdx.x;

    // Input: only lane 0 has 1
    int input = (lane == 0) ? 1 : 0;
    input_out[lane] = input;

    int result;
    // Pure row_shl:4 without bank_mask
    asm volatile(
        "s_nop 1\n"
        "v_mov_b32_dpp %0, %1 row_shl:4 row_mask:0xf bank_mask:0xf"
        : "=v"(result) : "v"(input) : "memory");
    shl_out[lane] = result;
}

int main() {
    int *d_input, *d_shl;
    int h_input[64], h_shl[64];

    hipMalloc(&d_input, 64 * sizeof(int));
    hipMalloc(&d_shl, 64 * sizeof(int));

    test_row_shl<<<1, 64>>>(d_input, d_shl);
    hipDeviceSynchronize();

    hipMemcpy(h_input, d_input, 64 * sizeof(int), hipMemcpyDeviceToHost);
    hipMemcpy(h_shl, d_shl, 64 * sizeof(int), hipMemcpyDeviceToHost);

    printf("row_shl:4 behavior test:\n");
    printf("Input (lane 0 = 1):\n  ");
    for (int i = 0; i < 16; i++) printf("%d", h_input[i]);
    printf("\n\n");

    printf("After row_shl:4 (bank_mask:0xf = all banks):\n  ");
    for (int i = 0; i < 16; i++) printf("%d", h_shl[i]);
    printf("\n\n");

    printf("row_shl:4 means: lane i gets value from lane (i-4)\n");
    printf("  Lane 0 gets from lane -4 (undefined/zero)\n");
    printf("  Lane 4 gets from lane 0 = 1\n");
    printf("  Lane 5 gets from lane 1 = 0\n");

    hipFree(d_input);
    hipFree(d_shl);
    return 0;
}
