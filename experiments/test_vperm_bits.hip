// Test: Does v_perm use 3-bit or 4-bit selectors?
#include <hip/hip_runtime.h>
#include <stdio.h>

__global__ void test_vperm_selector(uint32_t* output) {
    // 16-byte table in 4 uint32_t
    int8_t table[16] = {0, 1, 2, 3, 4, 6, 8, 12, 100, 101, 102, 103, 104, 106, 108, 112};
    uint32_t* t = (uint32_t*)table;

    // Test selector with 4-bit indices (8-15)
    // If v_perm uses 4 bits, we should get 100,101,102,103
    // If v_perm uses 3 bits, we should get 0,1,2,3 (wraps around)

    uint32_t sel = 0x0B0A0908;  // Bytes: 8,9,10,11
    uint32_t result = __builtin_amdgcn_perm(t[1], t[0], sel);  // Only first 8 bytes

    output[0] = result;

    // What about using all 4 uint32_t?
    // v_perm can only take 2 src operands (8 bytes), so this won't help
}

__global__ void test_lookup_simple(const int* q4_data, int2* output, const int8_t* table, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    uint32_t* t = (uint32_t*)table;
    int q4 = q4_data[idx];

    // User's proposed "simple" lookup
    uint32_t res_x = __builtin_amdgcn_perm(t[1], t[0], q4);
    uint32_t res_y = __builtin_amdgcn_perm(t[1], t[0], q4 >> 4);

    output[idx] = make_int2(res_x, res_y);
}

__global__ void test_baseline(const int* q4_data, int2* output, const int8_t* table, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    uint32_t* values = (uint32_t*)table;
    int q4 = q4_data[idx];

    uint32_t q_even = q4;
    uint32_t q_odd  = (q4 >> 4);

    uint32_t v_even_low = __builtin_amdgcn_perm(values[1], values[0], q_even & 0x07070707);
    uint32_t v_odd_low = __builtin_amdgcn_perm(values[1], values[0], q_odd & 0x07070707);
    uint32_t v_even_high = __builtin_amdgcn_perm(values[3], values[2], q_even & 0x07070707);
    uint32_t v_odd_high = __builtin_amdgcn_perm(values[3], values[2], q_odd & 0x07070707);

    uint32_t mask_even = 0x03020100 | ((q_even & 0x08080808) >> 1);
    uint32_t res_x = __builtin_amdgcn_perm(v_even_high, v_even_low, mask_even);
    uint32_t mask_odd = 0x03020100 | ((q_odd & 0x08080808) >> 1);
    uint32_t res_y = __builtin_amdgcn_perm(v_odd_high, v_odd_low, mask_odd);

    output[idx] = make_int2(res_x, res_y);
}

int main() {
    // First: test if v_perm uses 3 or 4 bits
    uint32_t* d_out;
    uint32_t h_out;
    hipMalloc(&d_out, sizeof(uint32_t));

    test_vperm_selector<<<1,1>>>(d_out);
    hipMemcpy(&h_out, d_out, sizeof(uint32_t), hipMemcpyDeviceToHost);

    printf("v_perm selector test (sel=0x0B0A0908):\n");
    printf("  Result: 0x%08x\n", h_out);
    printf("  If 4-bit: expect bytes 100,101,102,103 = 0x67666564\n");
    printf("  If 3-bit: expect bytes 0,1,2,3 = 0x03020100\n");

    uint8_t* bytes = (uint8_t*)&h_out;
    printf("  Got bytes: %d, %d, %d, %d\n", bytes[0], bytes[1], bytes[2], bytes[3]);

    if (bytes[0] == 0 && bytes[1] == 1) {
        printf("  => v_perm uses 3-BIT selectors (wraps at 8)\n");
    } else if (bytes[0] == 100 && bytes[1] == 101) {
        printf("  => v_perm uses 4-BIT selectors\n");
    }

    hipFree(d_out);

    // Second: compare lookup_simple vs baseline
    printf("\nComparing lookup_simple vs baseline:\n");

    const int N = 16;
    int8_t h_table[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};

    int* d_q4;
    int2* d_output;
    int8_t* d_table;

    hipMalloc(&d_q4, N * sizeof(int));
    hipMalloc(&d_output, N * sizeof(int2));
    hipMalloc(&d_table, 16);
    hipMemcpy(d_table, h_table, 16, hipMemcpyHostToDevice);

    // Test specific patterns
    int h_q4[16];
    for (int i = 0; i < 16; i++) {
        h_q4[i] = i | (i << 8) | (i << 16) | (i << 24);  // All bytes same nibble
    }
    hipMemcpy(d_q4, h_q4, N * sizeof(int), hipMemcpyHostToDevice);

    int2 h_baseline[16], h_simple[16];

    test_baseline<<<1, N>>>(d_q4, d_output, d_table, N);
    hipMemcpy(h_baseline, d_output, N * sizeof(int2), hipMemcpyDeviceToHost);

    test_lookup_simple<<<1, N>>>(d_q4, d_output, d_table, N);
    hipMemcpy(h_simple, d_output, N * sizeof(int2), hipMemcpyDeviceToHost);

    printf("nibble | baseline.x | simple.x | match?\n");
    for (int i = 0; i < 16; i++) {
        bool match = (h_baseline[i].x == h_simple[i].x);
        int8_t expected = h_table[i];
        printf("  %2d   | 0x%08x |0x%08x | %s (expect %d)\n",
               i, h_baseline[i].x, h_simple[i].x, match ? "YES" : "NO", expected);
    }

    hipFree(d_q4);
    hipFree(d_output);
    hipFree(d_table);

    return 0;
}
