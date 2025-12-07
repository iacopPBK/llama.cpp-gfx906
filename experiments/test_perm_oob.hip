// Test v_perm_b32 OOB behavior on GFX906
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>

__global__ void test_perm_oob(uint32_t* out) {
    if (threadIdx.x != 0) return;

    constexpr uint32_t src1 = 0x44332211u;  // bytes 4-7 = {0x11, 0x22, 0x33, 0x44}
    constexpr uint32_t src2 = 0x88776655u;  // bytes 0-3 = {0x55, 0x66, 0x77, 0x88}

    // Test various selector values
    uint32_t selectors[] = {
        0x00000000,  // All select byte 0 from src2
        0x07070707,  // All select byte 7 from src1
        0x80808080,  // All OOB (bit 7 set)
        0x81818181,  // All OOB (bit 7 set)
        0xFF000000,  // byte3 OOB, others valid
        0x03020100,  // identity
        0x07060504,  // upper 4 bytes
    };

    for (int i = 0; i < 7; i++) {
        uint32_t result;
        asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(result) : "v"(src1), "v"(src2), "v"(selectors[i]));
        out[i*3 + 0] = selectors[i];
        out[i*3 + 1] = result;
        out[i*3 + 2] = 0;  // padding
    }

    // Additional tests with different patterns
    // What does selector byte value > 7 but < 128 do?
    uint32_t test_sel = 0x0F0E0D0C;  // bytes 12,13,14,15 - should be invalid?
    uint32_t test_result;
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(test_result) : "v"(src1), "v"(src2), "v"(test_sel));
    out[21] = test_sel;
    out[22] = test_result;
    out[23] = 0;

    // What about the special values 0x80-0x87?
    // According to some docs, these might select the MSB of src0
    test_sel = 0x83828180;
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(test_result) : "v"(src1), "v"(src2), "v"(test_sel));
    out[24] = test_sel;
    out[25] = test_result;
    out[26] = 0;

    // Test 0xC0-0xC7 (select from src1 with bit 6 set?)
    test_sel = 0xC3C2C1C0;
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(test_result) : "v"(src1), "v"(src2), "v"(test_sel));
    out[27] = test_sel;
    out[28] = test_result;
    out[29] = 0;
}

int main() {
    uint32_t* d_out;
    hipMalloc(&d_out, 30 * sizeof(uint32_t));
    test_perm_oob<<<1, 1>>>(d_out);
    hipDeviceSynchronize();

    uint32_t h_out[30];
    hipMemcpy(h_out, d_out, 30 * sizeof(uint32_t), hipMemcpyDeviceToHost);

    printf("v_perm_b32 OOB Behavior Test on GFX906\n");
    printf("======================================\n\n");
    printf("src1 = 0x44332211 (bytes 4-7: 0x11, 0x22, 0x33, 0x44)\n");
    printf("src2 = 0x88776655 (bytes 0-3: 0x55, 0x66, 0x77, 0x88)\n\n");

    const char* labels[] = {
        "0x00000000 (all byte 0)",
        "0x07070707 (all byte 7)",
        "0x80808080 (all OOB bit7)",
        "0x81818181 (all OOB bit7)",
        "0xFF000000 (byte3 OOB)",
        "0x03020100 (identity)",
        "0x07060504 (upper bytes)",
        "0x0F0E0D0C (bytes 12-15?)",
        "0x83828180 (bit7 + bits)",
        "0xC3C2C1C0 (bits 6+7)"
    };

    for (int i = 0; i < 10; i++) {
        printf("Selector %s:\n", labels[i]);
        printf("  Result = 0x%08X (bytes: %02X %02X %02X %02X)\n\n",
               h_out[i*3 + 1],
               h_out[i*3 + 1] & 0xFF,
               (h_out[i*3 + 1] >> 8) & 0xFF,
               (h_out[i*3 + 1] >> 16) & 0xFF,
               (h_out[i*3 + 1] >> 24) & 0xFF);
    }

    hipFree(d_out);
    return 0;
}
