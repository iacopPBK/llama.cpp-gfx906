// Test ALL DPP operations used in warp_reduce to verify 32-lane isolation
// Compile: hipcc -o test_all_dpp test_all_dpp.cu --offload-arch=gfx906
// Run: ./test_all_dpp

#include <hip/hip_runtime.h>
#include <stdio.h>

// XOR 1: quad_perm:[1,0,3,2]
__device__ int dpp_xor1(int value) {
    int result;
    asm volatile(
        "s_nop 4\n"
        "v_mov_b32_dpp %0, %1 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf"
        : "=v"(result) : "v"(value) : "memory"
    );
    return result;
}

// XOR 2: quad_perm:[2,3,0,1]
__device__ int dpp_xor2(int value) {
    int result;
    asm volatile(
        "s_nop 1\n"
        "v_mov_b32_dpp %0, %1 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf"
        : "=v"(result) : "v"(value) : "memory"
    );
    return result;
}

// XOR 4: row_shl:4 + row_shr:4
__device__ int dpp_xor4(int value) {
    int result;
    asm volatile(
        "v_mov_b32 %0, %1\n"
        "s_nop 1\n"
        "v_mov_b32_dpp %0, %1 row_shl:4 row_mask:0xf bank_mask:0x5\n"
        "v_mov_b32_dpp %0, %1 row_shr:4 row_mask:0xf bank_mask:0xa\n"
        : "=v"(result) : "v"(value) : "memory"
    );
    return result;
}

// XOR 8: row_ror:8
__device__ int dpp_xor8(int value) {
    int result;
    asm volatile(
        "s_nop 1\n"
        "v_mov_b32_dpp %0, %1 row_ror:8 row_mask:0xf bank_mask:0xf"
        : "=v"(result) : "v"(value) : "memory"
    );
    return result;
}

// XOR 16: ds_swizzle SWAP,16
__device__ int dpp_xor16(int value) {
    int result;
    asm volatile(
        "ds_swizzle_b32 %0, %1 offset:swizzle(SWAP,16)\n"
        "s_waitcnt lgkmcnt(0)\n"
        : "=v"(result) : "v"(value) : "memory"
    );
    return result;
}

__global__ void test_dpp_ops(int* out_xor1, int* out_xor2, int* out_xor4, int* out_xor8, int* out_xor16) {
    int lane_id = threadIdx.x;

    out_xor1[lane_id] = dpp_xor1(lane_id);
    out_xor2[lane_id] = dpp_xor2(lane_id);
    out_xor4[lane_id] = dpp_xor4(lane_id);
    out_xor8[lane_id] = dpp_xor8(lane_id);
    out_xor16[lane_id] = dpp_xor16(lane_id);
}

int check_isolation(const char* name, int* output) {
    int cross = 0;
    for (int i = 0; i < 32; i++) {
        if (output[i] >= 32) {
            printf("  %s: Lane %d received from Lane %d (CROSSES BOUNDARY)\n", name, i, output[i]);
            cross++;
        }
    }
    for (int i = 32; i < 64; i++) {
        if (output[i] < 32) {
            printf("  %s: Lane %d received from Lane %d (CROSSES BOUNDARY)\n", name, i, output[i]);
            cross++;
        }
    }
    return cross;
}

int main() {
    int *d_xor1, *d_xor2, *d_xor4, *d_xor8, *d_xor16;
    int h_xor1[64], h_xor2[64], h_xor4[64], h_xor8[64], h_xor16[64];

    hipMalloc(&d_xor1, 64 * sizeof(int));
    hipMalloc(&d_xor2, 64 * sizeof(int));
    hipMalloc(&d_xor4, 64 * sizeof(int));
    hipMalloc(&d_xor8, 64 * sizeof(int));
    hipMalloc(&d_xor16, 64 * sizeof(int));

    test_dpp_ops<<<1, 64>>>(d_xor1, d_xor2, d_xor4, d_xor8, d_xor16);
    hipDeviceSynchronize();

    hipMemcpy(h_xor1, d_xor1, 64 * sizeof(int), hipMemcpyDeviceToHost);
    hipMemcpy(h_xor2, d_xor2, 64 * sizeof(int), hipMemcpyDeviceToHost);
    hipMemcpy(h_xor4, d_xor4, 64 * sizeof(int), hipMemcpyDeviceToHost);
    hipMemcpy(h_xor8, d_xor8, 64 * sizeof(int), hipMemcpyDeviceToHost);
    hipMemcpy(h_xor16, d_xor16, 64 * sizeof(int), hipMemcpyDeviceToHost);

    printf("Checking if DPP operations stay within 32-lane groups:\n");
    printf("======================================================\n\n");

    int total_cross = 0;
    total_cross += check_isolation("XOR1 (quad_perm)", h_xor1);
    total_cross += check_isolation("XOR2 (quad_perm)", h_xor2);
    total_cross += check_isolation("XOR4 (row_shl/shr)", h_xor4);
    total_cross += check_isolation("XOR8 (row_ror)", h_xor8);
    total_cross += check_isolation("XOR16 (ds_swizzle)", h_xor16);

    printf("\n======================================================\n");
    if (total_cross > 0) {
        printf("FAIL: %d cross-boundary communications detected!\n", total_cross);
    } else {
        printf("PASS: All DPP operations stay within 32-lane groups.\n");
        printf("\nThe bug in mm_ids_helper is NOT caused by DPP operations.\n");
        printf("Need to look elsewhere in the code.\n");
    }

    // Print actual mappings for verification
    printf("\n\nDetailed lane mappings (first 8 lanes of each group):\n");
    printf("Lane -> XOR1 XOR2 XOR4 XOR8 XOR16\n");
    for (int i = 0; i < 8; i++) {
        printf("%4d -> %4d %4d %4d %4d %5d\n", i, h_xor1[i], h_xor2[i], h_xor4[i], h_xor8[i], h_xor16[i]);
    }
    printf("...\n");
    for (int i = 32; i < 40; i++) {
        printf("%4d -> %4d %4d %4d %4d %5d\n", i, h_xor1[i], h_xor2[i], h_xor4[i], h_xor8[i], h_xor16[i]);
    }

    hipFree(d_xor1);
    hipFree(d_xor2);
    hipFree(d_xor4);
    hipFree(d_xor8);
    hipFree(d_xor16);

    return 0;
}
