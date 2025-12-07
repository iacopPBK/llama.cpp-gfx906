// Test XOR 4 with all lanes having their lane ID as value

#include <hip/hip_runtime.h>
#include <stdio.h>

__device__ int dpp_xor4(int value) {
    int result;
    asm volatile(
        "v_mov_b32 %0, %1\n"
        "s_nop 1\n"
        "v_mov_b32_dpp %0, %1 row_shl:4 row_mask:0xf bank_mask:0x5\n"
        "v_mov_b32_dpp %0, %1 row_shr:4 row_mask:0xf bank_mask:0xa\n"
        : "=v"(result) : "v"(value) : "memory");
    return result;
}

__global__ void test(int* out_laneids, int* out_xor4) {
    int lane = threadIdx.x;
    out_laneids[lane] = lane;
    out_xor4[lane] = dpp_xor4(lane);
}

int main() {
    int *d_lanes, *d_xor4;
    int h_lanes[64], h_xor4[64];

    hipMalloc(&d_lanes, 64 * sizeof(int));
    hipMalloc(&d_xor4, 64 * sizeof(int));

    test<<<1, 64>>>(d_lanes, d_xor4);
    hipDeviceSynchronize();

    hipMemcpy(h_lanes, d_lanes, 64 * sizeof(int), hipMemcpyDeviceToHost);
    hipMemcpy(h_xor4, d_xor4, 64 * sizeof(int), hipMemcpyDeviceToHost);

    printf("XOR 4 with lane IDs as input:\n");
    printf("Lane -> Input -> XOR4 result (expected: lane XOR 4)\n");
    for (int i = 0; i < 16; i++) {
        int expected = i ^ 4;
        printf("%4d -> %5d -> %4d (expected %d) %s\n",
               i, h_lanes[i], h_xor4[i], expected,
               h_xor4[i] == expected ? "OK" : "WRONG");
    }

    hipFree(d_lanes);
    hipFree(d_xor4);
    return 0;
}
