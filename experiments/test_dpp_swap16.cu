// Test to verify ds_swizzle SWAP,16 behavior on 64-wide wavefront
// Compile: hipcc -o test_dpp_swap16 test_dpp_swap16.cu --offload-arch=gfx906
// Run: ./test_dpp_swap16

#include <hip/hip_runtime.h>
#include <stdio.h>

__device__ int ds_swizzle_swap16(int value) {
    int result;
    asm volatile(
        "ds_swizzle_b32 %0, %1 offset:swizzle(SWAP,16)\n"
        "s_waitcnt lgkmcnt(0)\n"
        : "=v"(result) : "v"(value) : "memory"
    );
    return result;
}

__global__ void test_swap16(int* output) {
    int lane_id = threadIdx.x;  // 0-63

    // Each lane writes its own ID
    int my_value = lane_id;

    // Do the swap
    int swapped = ds_swizzle_swap16(my_value);

    // Output: for each lane, what lane did it receive from?
    output[lane_id] = swapped;
}

int main() {
    int* d_output;
    int h_output[64];

    hipMalloc(&d_output, 64 * sizeof(int));

    // Launch with exactly 64 threads (one wavefront)
    test_swap16<<<1, 64>>>(d_output);
    hipDeviceSynchronize();

    hipMemcpy(h_output, d_output, 64 * sizeof(int), hipMemcpyDeviceToHost);

    printf("ds_swizzle SWAP,16 lane mapping:\n");
    printf("================================\n");

    printf("\nGroup 0 (lanes 0-31):\n");
    for (int i = 0; i < 32; i++) {
        printf("  Lane %2d received from Lane %2d", i, h_output[i]);
        if (h_output[i] >= 32) {
            printf(" <-- CROSSES GROUP BOUNDARY!");
        }
        printf("\n");
    }

    printf("\nGroup 1 (lanes 32-63):\n");
    for (int i = 32; i < 64; i++) {
        printf("  Lane %2d received from Lane %2d", i, h_output[i]);
        if (h_output[i] < 32) {
            printf(" <-- CROSSES GROUP BOUNDARY!");
        }
        printf("\n");
    }

    // Check if any cross-group communication happened
    int cross_group = 0;
    for (int i = 0; i < 32; i++) {
        if (h_output[i] >= 32) cross_group++;
    }
    for (int i = 32; i < 64; i++) {
        if (h_output[i] < 32) cross_group++;
    }

    printf("\n================================\n");
    if (cross_group > 0) {
        printf("RESULT: SWAP,16 CROSSES 32-lane group boundaries! (%d lanes)\n", cross_group);
        printf("        This explains why warp_reduce_any<32> fails!\n");
    } else {
        printf("RESULT: SWAP,16 stays within 32-lane groups.\n");
        printf("        The bug must be elsewhere.\n");
    }

    hipFree(d_output);
    return 0;
}
