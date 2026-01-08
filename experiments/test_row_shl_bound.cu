// Test row_shl:4 with bound_ctrl behavior
// On AMD, when source lane is out of bounds, bound_ctrl determines behavior

#include <hip/hip_runtime.h>
#include <stdio.h>

__global__ void test_shl_variants(int* input_arr, int* shl_nobound, int* shl_bound0) {
    int lane = threadIdx.x;

    // Input: only lane 0 has 1
    int input = (lane == 0) ? 1 : 0;
    input_arr[lane] = input;

    int result1, result2;

    // Default behavior (no bound_ctrl specified - should preserve destination)
    result1 = input;  // Initialize to input value
    asm volatile(
        "s_nop 1\n"
        "v_mov_b32_dpp %0, %1 row_shl:4 row_mask:0xf bank_mask:0xf"
        : "+v"(result1) : "v"(input) : "memory");
    shl_nobound[lane] = result1;

    // With bound_ctrl:0 (explicitly fill with zero for out-of-bounds)
    result2 = input;
    asm volatile(
        "s_nop 1\n"
        "v_mov_b32_dpp %0, %1 row_shl:4 row_mask:0xf bank_mask:0xf bound_ctrl:0"
        : "+v"(result2) : "v"(input) : "memory");
    shl_bound0[lane] = result2;
}

int main() {
    int *d_input, *d_shl1, *d_shl2;
    int h_input[64], h_shl1[64], h_shl2[64];

    hipMalloc(&d_input, 64 * sizeof(int));
    hipMalloc(&d_shl1, 64 * sizeof(int));
    hipMalloc(&d_shl2, 64 * sizeof(int));

    test_shl_variants<<<1, 64>>>(d_input, d_shl1, d_shl2);
    hipDeviceSynchronize();

    hipMemcpy(h_input, d_input, 64 * sizeof(int), hipMemcpyDeviceToHost);
    hipMemcpy(h_shl1, d_shl1, 64 * sizeof(int), hipMemcpyDeviceToHost);
    hipMemcpy(h_shl2, d_shl2, 64 * sizeof(int), hipMemcpyDeviceToHost);

    printf("Input (first 16 lanes):\n  ");
    for (int i = 0; i < 16; i++) printf("%d", h_input[i]);
    printf("\n\n");

    printf("After row_shl:4 (default, +v constraint):\n  ");
    for (int i = 0; i < 16; i++) printf("%d", h_shl1[i]);
    printf("\n");

    printf("After row_shl:4 bound_ctrl:0:\n  ");
    for (int i = 0; i < 16; i++) printf("%d", h_shl2[i]);
    printf("\n\n");

    printf("Expected: lane 4 should have 1 (from lane 0)\n");
    printf("Lane 4: shl1=%d, shl2=%d\n", h_shl1[4], h_shl2[4]);

    hipFree(d_input);
    hipFree(d_shl1);
    hipFree(d_shl2);
    return 0;
}
