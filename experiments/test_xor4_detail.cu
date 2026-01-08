// Detailed test of XOR 4 behavior

#include <hip/hip_runtime.h>
#include <stdio.h>

// Your current XOR 4 implementation
__device__ int dpp_xor4_current(int value) {
    int result;
    asm volatile(
        "v_mov_b32 %0, %1\n"
        "s_nop 1\n"
        "v_mov_b32_dpp %0, %1 row_shl:4 row_mask:0xf bank_mask:0x5\n"
        "v_mov_b32_dpp %0, %1 row_shr:4 row_mask:0xf bank_mask:0xa\n"
        : "=v"(result) : "v"(value) : "memory");
    return result;
}

__global__ void test_xor4(int* input_out, int* shuffle_out) {
    int lane = threadIdx.x;

    // Only lane 0 has value 1
    int x = (lane == 0) ? 1 : 0;
    input_out[lane] = x;

    int shuffled = dpp_xor4_current(x);
    shuffle_out[lane] = shuffled;
}

int main() {
    int *d_input, *d_shuffle;
    int h_input[64], h_shuffle[64];

    hipMalloc(&d_input, 64 * sizeof(int));
    hipMalloc(&d_shuffle, 64 * sizeof(int));

    test_xor4<<<1, 64>>>(d_input, d_shuffle);
    hipDeviceSynchronize();

    hipMemcpy(h_input, d_input, 64 * sizeof(int), hipMemcpyDeviceToHost);
    hipMemcpy(h_shuffle, d_shuffle, 64 * sizeof(int), hipMemcpyDeviceToHost);

    printf("XOR 4 detailed test:\n");
    printf("Input (lane 0 = 1, others = 0):\n");
    printf("  Row 0 (lanes 0-15):  ");
    for (int i = 0; i < 16; i++) printf("%d", h_input[i]);
    printf("\n");

    printf("\nAfter dpp_xor4 (should be XOR with lane+4):\n");
    printf("  Row 0 (lanes 0-15):  ");
    for (int i = 0; i < 16; i++) printf("%d", h_shuffle[i]);
    printf("\n");

    printf("\nExpected for XOR 4 shuffle:\n");
    printf("  Lane 0 should get value from lane 4 (0)\n");
    printf("  Lane 4 should get value from lane 0 (1)\n");
    printf("  For reduction: result[i] = shuffle[i] || input[i]\n");
    printf("    So lane 0 should be: 0 || 1 = 1\n");
    printf("    And lane 4 should be: 1 || 0 = 1\n");

    printf("\nActual shuffle result at lane 0: %d (expected: value from lane 4 = 0)\n", h_shuffle[0]);
    printf("Actual shuffle result at lane 4: %d (expected: value from lane 0 = 1)\n", h_shuffle[4]);

    hipFree(d_input);
    hipFree(d_shuffle);
    return 0;
}
