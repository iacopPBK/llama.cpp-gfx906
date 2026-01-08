// Step through XOR 4 DPP implementation

#include <hip/hip_runtime.h>
#include <stdio.h>

__global__ void test_steps(int* after_mov, int* after_shl, int* after_shr) {
    int lane = threadIdx.x;
    int input = (lane == 0) ? 1 : 0;

    int result;

    // Step 1: v_mov_b32
    asm volatile("v_mov_b32 %0, %1" : "=v"(result) : "v"(input));
    after_mov[lane] = result;

    // Step 2: row_shl:4 with bank_mask:0x5
    asm volatile(
        "s_nop 1\n"
        "v_mov_b32_dpp %0, %1 row_shl:4 row_mask:0xf bank_mask:0x5"
        : "+v"(result) : "v"(input) : "memory");
    after_shl[lane] = result;

    // Step 3: row_shr:4 with bank_mask:0xa
    asm volatile(
        "v_mov_b32_dpp %0, %1 row_shr:4 row_mask:0xf bank_mask:0xa"
        : "+v"(result) : "v"(input) : "memory");
    after_shr[lane] = result;
}

int main() {
    int *d_mov, *d_shl, *d_shr;
    int h_mov[64], h_shl[64], h_shr[64];

    hipMalloc(&d_mov, 64 * sizeof(int));
    hipMalloc(&d_shl, 64 * sizeof(int));
    hipMalloc(&d_shr, 64 * sizeof(int));

    test_steps<<<1, 64>>>(d_mov, d_shl, d_shr);
    hipDeviceSynchronize();

    hipMemcpy(h_mov, d_mov, 64 * sizeof(int), hipMemcpyDeviceToHost);
    hipMemcpy(h_shl, d_shl, 64 * sizeof(int), hipMemcpyDeviceToHost);
    hipMemcpy(h_shr, d_shr, 64 * sizeof(int), hipMemcpyDeviceToHost);

    printf("Input: only lane 0 = 1\n\n");

    printf("After v_mov_b32 (first 16 lanes):\n  ");
    for (int i = 0; i < 16; i++) printf("%d", h_mov[i]);
    printf("\n\n");

    printf("After row_shl:4 bank_mask:0x5 (first 16 lanes):\n  ");
    for (int i = 0; i < 16; i++) printf("%d", h_shl[i]);
    printf("\n");
    printf("  Banks 0,2 active. Lane 4 (bank 0) should get 1 from lane 0.\n\n");

    printf("After row_shr:4 bank_mask:0xa (first 16 lanes):\n  ");
    for (int i = 0; i < 16; i++) printf("%d", h_shr[i]);
    printf("\n");
    printf("  Banks 1,3 active. Should not change lanes in banks 0,2.\n\n");

    printf("Bank assignments for lanes 0-7:\n");
    for (int i = 0; i < 8; i++) {
        printf("  Lane %d: bank %d\n", i, i % 4);
    }

    hipFree(d_mov);
    hipFree(d_shl);
    hipFree(d_shr);
    return 0;
}
