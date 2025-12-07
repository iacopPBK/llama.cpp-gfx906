// Test XOR 4 with sparse input (only lane 0 has value)

#include <hip/hip_runtime.h>
#include <stdio.h>

__device__ int xor4_current(int value) {
    int result;
    asm volatile(
        "v_mov_b32 %0, %1\n"
        "s_nop 1\n"
        "v_mov_b32_dpp %0, %1 row_shl:4 row_mask:0xf bank_mask:0x5\n"
        "v_mov_b32_dpp %0, %1 row_shr:4 row_mask:0xf bank_mask:0xa\n"
        : "=v"(result) : "v"(value) : "memory");
    return result;
}

__device__ int xor4_shfl(int value) {
    return __shfl_xor(value, 4, 64);
}

__global__ void test_sparse(int* out_current, int* out_shfl) {
    int lane = threadIdx.x;

    // Sparse input: only lane 0 has value 1
    int input = (lane == 0) ? 1 : 0;

    out_current[lane] = xor4_current(input);
    out_shfl[lane] = xor4_shfl(input);
}

__global__ void test_dense(int* out_current, int* out_shfl) {
    int lane = threadIdx.x;

    // Dense input: lane ID as value
    int input = lane;

    out_current[lane] = xor4_current(input);
    out_shfl[lane] = xor4_shfl(input);
}

int main() {
    int *d_current, *d_shfl;
    int h_current[64], h_shfl[64];

    hipMalloc(&d_current, 64 * sizeof(int));
    hipMalloc(&d_shfl, 64 * sizeof(int));

    printf("=== Test with SPARSE input (only lane 0 = 1) ===\n");
    test_sparse<<<1, 64>>>(d_current, d_shfl);
    hipDeviceSynchronize();

    hipMemcpy(h_current, d_current, 64 * sizeof(int), hipMemcpyDeviceToHost);
    hipMemcpy(h_shfl, d_shfl, 64 * sizeof(int), hipMemcpyDeviceToHost);

    printf("Current DPP (first 16): ");
    for (int i = 0; i < 16; i++) printf("%d", h_current[i]);
    printf("\n");
    printf("__shfl_xor (first 16):  ");
    for (int i = 0; i < 16; i++) printf("%d", h_shfl[i]);
    printf("\n");
    printf("Expected (lane 4 = 1):  0000100000000000\n\n");

    printf("Lane 0 value: current=%d, shfl=%d (expected 0, got from lane 4)\n", h_current[0], h_shfl[0]);
    printf("Lane 4 value: current=%d, shfl=%d (expected 1, got from lane 0)\n", h_current[4], h_shfl[4]);

    printf("\n=== Test with DENSE input (lane ID as value) ===\n");
    test_dense<<<1, 64>>>(d_current, d_shfl);
    hipDeviceSynchronize();

    hipMemcpy(h_current, d_current, 64 * sizeof(int), hipMemcpyDeviceToHost);
    hipMemcpy(h_shfl, d_shfl, 64 * sizeof(int), hipMemcpyDeviceToHost);

    printf("Current DPP (first 16): ");
    for (int i = 0; i < 16; i++) printf("%2d ", h_current[i]);
    printf("\n");
    printf("__shfl_xor (first 16):  ");
    for (int i = 0; i < 16; i++) printf("%2d ", h_shfl[i]);
    printf("\n");
    printf("Expected (i XOR 4):     ");
    for (int i = 0; i < 16; i++) printf("%2d ", i ^ 4);
    printf("\n");

    hipFree(d_current);
    hipFree(d_shfl);
    return 0;
}
