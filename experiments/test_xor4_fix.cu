// Test different XOR 4 implementations

#include <hip/hip_runtime.h>
#include <stdio.h>

// Method 1: Your current broken implementation
__device__ int xor4_broken(int value) {
    int result;
    asm volatile(
        "v_mov_b32 %0, %1\n"
        "s_nop 1\n"
        "v_mov_b32_dpp %0, %1 row_shl:4 row_mask:0xf bank_mask:0x5\n"
        "v_mov_b32_dpp %0, %1 row_shr:4 row_mask:0xf bank_mask:0xa\n"
        : "=v"(result) : "v"(value) : "memory");
    return result;
}

// Method 2: Using row_xmask (if supported on gfx906)
__device__ int xor4_xmask(int value) {
    int result;
    asm volatile(
        "s_nop 1\n"
        "v_mov_b32_dpp %0, %1 row_xmask:4 row_mask:0xf bank_mask:0xf"
        : "=v"(result) : "v"(value) : "memory");
    return result;
}

// Method 3: Using native shuffle
__device__ int xor4_shfl(int value) {
    return __shfl_xor(value, 4, 64);
}

// Method 4: Using ds_swizzle with XOR pattern
__device__ int xor4_swizzle(int value) {
    int result;
    // ds_swizzle BitMode: bit15=1, bits[14:10]=and_mask, bits[9:5]=or_mask, bits[4:0]=xor_mask
    // For XOR 4: and_mask=0x1f (keep all), or_mask=0x00, xor_mask=0x04
    // Pattern = 0x8000 | (0x1f << 10) | (0x00 << 5) | 0x04 = 0x8000 | 0x7C00 | 0x04 = 0xFC04
    asm volatile(
        "ds_swizzle_b32 %0, %1 offset:0xFC04\n"
        "s_waitcnt lgkmcnt(0)\n"
        : "=v"(result) : "v"(value) : "memory");
    return result;
}

// Method 5: Two separate DPP operations with proper lane selection
__device__ int xor4_two_dpp(int value) {
    int lo_result, hi_result;
    // For lanes 0-3, 8-11 (need value from +4): use row_shr:4
    // For lanes 4-7, 12-15 (need value from -4): use row_shl:4

    asm volatile(
        "s_nop 1\n"
        "v_mov_b32_dpp %0, %1 row_shr:4 row_mask:0xf bank_mask:0xf"
        : "=v"(lo_result) : "v"(value) : "memory");

    asm volatile(
        "s_nop 1\n"
        "v_mov_b32_dpp %0, %1 row_shl:4 row_mask:0xf bank_mask:0xf"
        : "=v"(hi_result) : "v"(value) : "memory");

    // Select based on lane position within group of 8
    int lane = threadIdx.x;
    int pos_in_8 = lane & 0x7;  // Position within each 8-lane group
    return (pos_in_8 < 4) ? lo_result : hi_result;
}

__global__ void test_all(int* out_broken, int* out_xmask, int* out_shfl,
                         int* out_swizzle, int* out_two_dpp) {
    int lane = threadIdx.x;

    out_broken[lane] = xor4_broken(lane);
    // out_xmask[lane] = xor4_xmask(lane);  // May not compile on gfx906
    out_shfl[lane] = xor4_shfl(lane);
    out_swizzle[lane] = xor4_swizzle(lane);
    out_two_dpp[lane] = xor4_two_dpp(lane);
}

void check_result(const char* name, int* data) {
    int errors = 0;
    for (int i = 0; i < 16; i++) {
        int expected = i ^ 4;
        if (data[i] != expected) errors++;
    }
    printf("%s: %s (%d errors in first 16 lanes)\n",
           name, errors == 0 ? "CORRECT" : "BROKEN", errors);

    if (errors > 0) {
        printf("  First 16 lanes: ");
        for (int i = 0; i < 16; i++) printf("%d ", data[i]);
        printf("\n  Expected:       ");
        for (int i = 0; i < 16; i++) printf("%d ", i ^ 4);
        printf("\n");
    }
}

int main() {
    int *d_broken, *d_xmask, *d_shfl, *d_swizzle, *d_two_dpp;
    int h_broken[64], h_xmask[64], h_shfl[64], h_swizzle[64], h_two_dpp[64];

    hipMalloc(&d_broken, 64 * sizeof(int));
    hipMalloc(&d_xmask, 64 * sizeof(int));
    hipMalloc(&d_shfl, 64 * sizeof(int));
    hipMalloc(&d_swizzle, 64 * sizeof(int));
    hipMalloc(&d_two_dpp, 64 * sizeof(int));

    test_all<<<1, 64>>>(d_broken, d_xmask, d_shfl, d_swizzle, d_two_dpp);
    hipDeviceSynchronize();

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        printf("Kernel error: %s\n", hipGetErrorString(err));
    }

    hipMemcpy(h_broken, d_broken, 64 * sizeof(int), hipMemcpyDeviceToHost);
    hipMemcpy(h_shfl, d_shfl, 64 * sizeof(int), hipMemcpyDeviceToHost);
    hipMemcpy(h_swizzle, d_swizzle, 64 * sizeof(int), hipMemcpyDeviceToHost);
    hipMemcpy(h_two_dpp, d_two_dpp, 64 * sizeof(int), hipMemcpyDeviceToHost);

    printf("Testing XOR 4 implementations:\n");
    printf("==============================\n");
    check_result("Broken (current)", h_broken);
    check_result("__shfl_xor", h_shfl);
    check_result("ds_swizzle BITMASK", h_swizzle);
    check_result("Two DPP + select", h_two_dpp);

    hipFree(d_broken);
    hipFree(d_xmask);
    hipFree(d_shfl);
    hipFree(d_swizzle);
    hipFree(d_two_dpp);
    return 0;
}
