// Test: MXFP4 lookup optimizations for GFX906
// Goal: Beat current 4 v_perm + BFI pattern
// Table: { 0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12 }
// Pattern: bit3 = sign, bits 0-2 = magnitude index from {0,1,2,3,4,6,8,12}

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>

// Table: { 0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12 }
__constant__ int8_t kvalues_mxfp4[16] = {
    0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12
};

// Magnitude table (8 entries, fits in one v_perm pair)
__constant__ uint8_t mags[8] = { 0, 1, 2, 3, 4, 6, 8, 12 };

#define ITERS 100000
#define BLOCK_SIZE 256

// ============================================================================
// BASELINE: Current 4-perm + BFI implementation (from vecdotq.cuh)
// ============================================================================
__device__ __forceinline__ int2 baseline_lookup(const int q4, const int8_t* table) {
    const uint32_t *values = (const uint32_t *)table;
    const uint32_t q_even = q4;
    const uint32_t q_odd  = (q4 >> 4);

    const uint32_t sel_even = q_even & 0x07070707;
    const uint32_t sel_odd  = q_odd & 0x07070707;

    // 4 v_perm for lookups
    uint32_t v_even_low = __builtin_amdgcn_perm(values[1], values[0], sel_even);
    uint32_t v_odd_low = __builtin_amdgcn_perm(values[1], values[0], sel_odd);
    uint32_t v_even_high = __builtin_amdgcn_perm(values[3], values[2], sel_even);
    uint32_t v_odd_high = __builtin_amdgcn_perm(values[3], values[2], sel_odd);

    // Expand bit3 to byte mask
    uint32_t b3e = (q_even >> 3) & 0x01010101;
    uint32_t me = b3e; me |= me << 1; me |= me << 2; me |= me << 4;
    uint32_t b3o = (q_odd >> 3) & 0x01010101;
    uint32_t mo = b3o; mo |= mo << 1; mo |= mo << 2; mo |= mo << 4;

    // BFI select: (high & mask) | (low & ~mask)
    uint32_t res_x = (v_even_high & me) | (v_even_low & ~me);
    uint32_t res_y = (v_odd_high & mo) | (v_odd_low & ~mo);
    return make_int2(res_x, res_y);
}

// ============================================================================
// OPT1: 2-perm + packed byte negate using v_sub_u8
// Problem: Can't negate packed bytes with single - operator
// Solution: For signed values, use 0 - x which is v_sub per byte
// ============================================================================
__device__ __forceinline__ int2 opt1_2perm_fixed(const int q4, const uint8_t* mag_table) {
    const uint32_t *mags32 = (const uint32_t *)mag_table;
    const uint32_t q_even = q4;
    const uint32_t q_odd  = (q4 >> 4);

    const uint32_t sel_even = q_even & 0x07070707;
    const uint32_t sel_odd  = q_odd & 0x07070707;

    // 2 v_perm for magnitude lookup
    uint32_t mag_even = __builtin_amdgcn_perm(mags32[1], mags32[0], sel_even);
    uint32_t mag_odd  = __builtin_amdgcn_perm(mags32[1], mags32[0], sel_odd);

    // Extract sign bits
    uint32_t sign_even = (q_even >> 3) & 0x01010101;
    uint32_t sign_odd  = (q_odd >> 3) & 0x01010101;

    // Create doubled magnitude for subtraction
    // If sign=1: result = 0 - mag = -mag (need packed byte subtract)
    // If sign=0: result = mag - 0 = mag

    // Use packed subtract: For each byte, compute (sign ? -mag : mag)
    // Packed negate: ~x + 1 doesn't work for packed bytes
    // Alternative: result = mag - 2*mag*sign = mag * (1 - 2*sign)
    //            But this needs multiplication

    // Simplest correct approach: compute both and select
    // neg_mag = ~mag + 0x01010101 (but carries between bytes!)

    // Actually correct approach: use separate byte handling
    // or use v_perm to select between +mag and -mag from table

    // NEW APPROACH: Use the full 16-entry table but structure lookup better
    // Table is {0,1,2,3,4,6,8,12} then {0,-1,-2,-3,-4,-6,-8,-12}
    // Index 0-7: positive, Index 8-15: negative

    // The original baseline already handles this optimally with 4 v_perm + BFI
    // We can try: 2 v_perm for each half, then use bit3 to select

    // Actually let's try: Since neg values are just -pos, and we have:
    // table[8+i] = -table[i] for i=0..7
    // We can: lookup(idx & 7), then negate if idx >= 8

    // Packed byte negate using XOR and packed add:
    // -x = ~x + 1 for single byte
    // For packed: (mag_even ^ 0xFFFFFFFF) + 0x01010101 ALMOST works
    //   but carries can propagate between bytes if mag >= 0x80

    // Since our magnitudes are max 12 = 0x0C, no overflow!
    uint32_t mask_even = sign_even * 0xFF;
    uint32_t mask_odd  = sign_odd * 0xFF;

    // XOR to flip bits, add 1 per byte for twos complement
    // Only add 1 where sign bit is set
    uint32_t res_x = (mag_even ^ mask_even) + sign_even;
    uint32_t res_y = (mag_odd ^ mask_odd) + sign_odd;

    return make_int2(res_x, res_y);
}

// Debug: Print byte values
__device__ void print_bytes(const char* name, uint32_t v) {
    printf("  %s: %02x %02x %02x %02x\n", name,
           (v >> 0) & 0xFF, (v >> 8) & 0xFF, (v >> 16) & 0xFF, (v >> 24) & 0xFF);
}

// ============================================================================
// Debug kernel to trace the computation
// ============================================================================
__global__ void debug_kernel() {
    // Test a few values
    int test_vals[] = {0x00, 0x08, 0x0F, 0x88, 0xFF};

    for (int t = 0; t < 5; t++) {
        int q4 = test_vals[t];
        printf("\nq4 = 0x%02x:\n", q4);

        int2 base = baseline_lookup(q4, kvalues_mxfp4);
        int2 opt1 = opt1_2perm_fixed(q4, mags);

        printf("  baseline: x=0x%08x y=0x%08x\n", base.x, base.y);
        printf("  opt1:     x=0x%08x y=0x%08x\n", opt1.x, opt1.y);
        printf("  match: %s\n", (base.x == opt1.x && base.y == opt1.y) ? "YES" : "NO");

        // Trace opt1 computation
        uint32_t q_even = q4;
        uint32_t sel_even = q_even & 0x07070707;
        uint32_t mag_even = __builtin_amdgcn_perm(((uint32_t*)mags)[1], ((uint32_t*)mags)[0], sel_even);
        uint32_t sign_even = (q_even >> 3) & 0x01010101;
        uint32_t mask_even = sign_even * 0xFF;
        printf("  sel_even=0x%08x mag_even=0x%08x sign=0x%08x mask=0x%08x\n",
               sel_even, mag_even, sign_even, mask_even);
    }
}

// ============================================================================
// KERNELS
// ============================================================================
__global__ void kernel_baseline(const int* input, int2* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int2 sum = make_int2(0, 0);
    for (int i = 0; i < ITERS; i++) {
        int2 v = baseline_lookup(input[idx % n], kvalues_mxfp4);
        sum.x += v.x; sum.y += v.y;
    }
    output[idx] = sum;
}

__global__ void kernel_opt1(const int* input, int2* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int2 sum = make_int2(0, 0);
    for (int i = 0; i < ITERS; i++) {
        int2 v = opt1_2perm_fixed(input[idx % n], mags);
        sum.x += v.x; sum.y += v.y;
    }
    output[idx] = sum;
}

// ============================================================================
// VERIFICATION
// ============================================================================
__global__ void verify_kernel(int* errors) {
    int idx = threadIdx.x;
    if (idx >= 256) return;

    // Test all 256 possible single-byte values replicated
    int q4 = idx | (idx << 8) | (idx << 16) | (idx << 24);

    int2 base = baseline_lookup(q4, kvalues_mxfp4);
    int2 opt1 = opt1_2perm_fixed(q4, mags);

    if (base.x != opt1.x || base.y != opt1.y) {
        atomicAdd(errors, 1);
        if (atomicAdd(errors + 1, 1) < 5) {  // Only print first 5 failures
            printf("FAIL at idx=%d (q4=0x%08x): base=(0x%08x,0x%08x) opt1=(0x%08x,0x%08x)\n",
                   idx, q4, base.x, base.y, opt1.x, opt1.y);
        }
    }
}

void verify() {
    int* d_errors;
    int h_errors[2] = {0, 0};

    hipMalloc(&d_errors, 2 * sizeof(int));
    hipMemset(d_errors, 0, 2 * sizeof(int));

    verify_kernel<<<1, 256>>>(d_errors);
    hipDeviceSynchronize();

    hipMemcpy(h_errors, d_errors, 2 * sizeof(int), hipMemcpyDeviceToHost);
    printf("Verification: %d/%d passed\n", 256 - h_errors[0], 256);

    hipFree(d_errors);
}

void bench(const char* name, void(*kernel)(const int*, int2*, int), const int* d_input, int2* d_output, int n) {
    kernel<<<1, BLOCK_SIZE>>>(d_input, d_output, n);
    hipDeviceSynchronize();

    hipEvent_t start, stop;
    hipEventCreate(&start); hipEventCreate(&stop);
    hipEventRecord(start);
    kernel<<<1, BLOCK_SIZE>>>(d_input, d_output, n);
    hipEventRecord(stop);
    hipEventSynchronize(stop);

    float ms;
    hipEventElapsedTime(&ms, start, stop);
    double lookups = (double)BLOCK_SIZE * ITERS;
    double gops = lookups / (ms * 1e6);
    printf("  %-30s: %.3f ms (%.2f G lookups/s)\n", name, ms, gops);

    hipEventDestroy(start);
    hipEventDestroy(stop);
}

int main() {
    printf("MXFP4 Lookup Optimization Test (GFX906)\n");
    printf("========================================\n\n");

    printf("Debug trace:\n");
    debug_kernel<<<1, 1>>>();
    hipDeviceSynchronize();
    printf("\n");

    verify();
    printf("\n");

    // Allocate
    int* d_input;
    int2* d_output;
    hipMalloc(&d_input, BLOCK_SIZE * sizeof(int));
    hipMalloc(&d_output, BLOCK_SIZE * sizeof(int2));

    // Init
    int h_input[BLOCK_SIZE];
    for (int i = 0; i < BLOCK_SIZE; i++) {
        h_input[i] = (i * 0x12345678) ^ (i << 16);
    }
    hipMemcpy(d_input, h_input, BLOCK_SIZE * sizeof(int), hipMemcpyHostToDevice);

    printf("Benchmarks (%d iters, %d threads):\n", ITERS, BLOCK_SIZE);
    bench("Baseline (4 perm + BFI)", kernel_baseline, d_input, d_output, BLOCK_SIZE);
    bench("Opt1 (2 perm + arith)", kernel_opt1, d_input, d_output, BLOCK_SIZE);

    hipFree(d_input);
    hipFree(d_output);
    return 0;
}
