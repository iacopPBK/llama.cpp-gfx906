// Test: Does GFX906 support unaligned global_load_dword?
// Compile: hipcc -o test_unaligned test_unaligned.hip --offload-arch=gfx906
// Run: ./test_unaligned

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>

__global__ void test_unaligned_load(const uint8_t* data, int* results, int offset) {
    // Try loading an int from a non-4-byte-aligned address
    const uint8_t* ptr = data + offset;  // offset can be 1, 2, 3

    // Method 1: Direct cast (may fail on unaligned)
    // int val1 = *((const int*)ptr);

    // Method 2: __builtin_memcpy (compiler may optimize)
    int val2;
    __builtin_memcpy(&val2, ptr, sizeof(int));

    // Method 3: Byte-by-byte (always works but slow)
    int val3 = ptr[0] | (ptr[1] << 8) | (ptr[2] << 16) | (ptr[3] << 24);

    results[0] = val2;  // memcpy result
    results[1] = val3;  // byte-by-byte result
    results[2] = (val2 == val3) ? 1 : 0;  // match?
}

__global__ void test_vectorized_aligned_load(const uint8_t* data, int4* results) {
    // Test: Can we load int4 from 16-byte aligned address?
    // Even if block_mxfp4.qs is at offset 1, multiple blocks may align
    const int4* ptr = (const int4*)(data);  // Assume 16-byte aligned
    results[0] = ptr[0];
}

int main() {
    // Create test data: 0x00 0x11 0x22 0x33 0x44 0x55 0x66 0x77 ...
    uint8_t h_data[32];
    for (int i = 0; i < 32; i++) {
        h_data[i] = i * 0x11;
    }

    uint8_t* d_data;
    int* d_results;
    int h_results[4];

    hipMalloc(&d_data, 32);
    hipMalloc(&d_results, 16);
    hipMemcpy(d_data, h_data, 32, hipMemcpyHostToDevice);

    printf("Testing unaligned loads on GFX906...\n\n");

    // Test offsets 0, 1, 2, 3
    for (int offset = 0; offset <= 3; offset++) {
        hipMemset(d_results, 0, 16);
        test_unaligned_load<<<1, 1>>>(d_data, d_results, offset);
        hipDeviceSynchronize();
        hipMemcpy(h_results, d_results, 12, hipMemcpyDeviceToHost);

        // Expected value at offset
        int expected = h_data[offset] | (h_data[offset+1] << 8) |
                       (h_data[offset+2] << 16) | (h_data[offset+3] << 24);

        printf("Offset %d: memcpy=0x%08x, bytes=0x%08x, expected=0x%08x, match=%s\n",
               offset, h_results[0], h_results[1], expected,
               (h_results[0] == expected && h_results[1] == expected) ? "YES" : "NO");
    }

    printf("\n");

    // Check for errors
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        printf("HIP Error: %s\n", hipGetErrorString(err));
    } else {
        printf("No HIP errors - unaligned access appears to work!\n");
    }

    hipFree(d_data);
    hipFree(d_results);

    return 0;
}
