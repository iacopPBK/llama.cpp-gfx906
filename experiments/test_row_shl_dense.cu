// Test row_shl:4 with dense input to understand the operation
// row_shl:4 should shift left by 4 within each row of 16 lanes

#include <hip/hip_runtime.h>
#include <stdio.h>

__global__ void test_row_shl_dense(int* input_arr, int* output_arr) {
    int lane = threadIdx.x;

    // Dense input: lane ID as value
    int input = lane;
    input_arr[lane] = input;

    int result;
    asm volatile(
        "v_mov_b32 %0, %1\n"
        "s_nop 1\n"
        "v_mov_b32_dpp %0, %1 row_shl:4 row_mask:0xf bank_mask:0xf"
        : "=v"(result) : "v"(input) : "memory");
    output_arr[lane] = result;
}

int main() {
    int *d_input, *d_output;
    int h_input[64], h_output[64];

    hipMalloc(&d_input, 64 * sizeof(int));
    hipMalloc(&d_output, 64 * sizeof(int));

    test_row_shl_dense<<<1, 64>>>(d_input, d_output);
    hipDeviceSynchronize();

    hipMemcpy(h_input, d_input, 64 * sizeof(int), hipMemcpyDeviceToHost);
    hipMemcpy(h_output, d_output, 64 * sizeof(int), hipMemcpyDeviceToHost);

    printf("row_shl:4 behavior with dense input (lane ID):\n\n");
    printf("Lane:  ");
    for (int i = 0; i < 16; i++) printf("%2d ", i);
    printf("\n");
    printf("Input: ");
    for (int i = 0; i < 16; i++) printf("%2d ", h_input[i]);
    printf("\n");
    printf("Output:");
    for (int i = 0; i < 16; i++) printf("%2d ", h_output[i]);
    printf("\n\n");

    printf("If row_shl:4 means 'shift left by 4', then:\n");
    printf("  Lane 0-3: get 0 (source lane negative)\n");
    printf("  Lane 4: gets value from lane 0 = 0\n");
    printf("  Lane 5: gets value from lane 1 = 1\n");
    printf("  etc.\n\n");

    printf("If row_shl:4 means 'lane gets value from lane-4':\n");
    printf("  Lane 4 gets from lane 0 = 0\n");
    printf("  Lane 5 gets from lane 1 = 1\n");
    printf("  etc.\n");

    hipFree(d_input);
    hipFree(d_output);
    return 0;
}
