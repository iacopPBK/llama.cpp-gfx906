// GPU microbenchmark: MXFP4 table lookup methods
// Compile: hipcc -O3 --offload-arch=gfx906 -o test_mxfp4_lds_bench test_mxfp4_lds_bench.hip
// Run: HSA_OVERRIDE_GFX_VERSION=9.0.6 ./test_mxfp4_lds_bench

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>

#define ITERATIONS 1000000
#define THREADS 256
#define BLOCKS 256

// Original table
__device__ __constant__ int8_t kvalues_mxfp4_const[16] = {
    0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12
};

// Method 1: Current v_perm implementation (6 perm ops)
static __device__ __forceinline__ int2 lookup_perm(const uint32_t q4, const int8_t * values) {
    uint32_t values_u32[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        values_u32[i] = reinterpret_cast<const uint32_t*>(values)[i];
    }

    uint32_t q_even = (q4 & 0x0f0f0f0f);
    uint32_t q_odd  = (q4 >> 4) & 0x0f0f0f0f;

    uint32_t mask_even = (q_even >> 3) * 0xff;
    uint32_t mask_odd  = (q_odd >> 3)  * 0xff;
    mask_even = (mask_even & 0x0c0c0c0c) | 0x03020100;
    mask_odd  = (mask_odd  & 0x0c0c0c0c) | 0x03020100;

    uint32_t v_even_low  = __builtin_amdgcn_perm(values_u32[1], values_u32[0], q_even & 0x07070707);
    uint32_t v_odd_low   = __builtin_amdgcn_perm(values_u32[1], values_u32[0], q_odd  & 0x07070707);
    uint32_t v_even_high = __builtin_amdgcn_perm(values_u32[3], values_u32[2], q_even & 0x07070707);
    uint32_t v_odd_high  = __builtin_amdgcn_perm(values_u32[3], values_u32[2], q_odd  & 0x07070707);
    uint32_t res_x = __builtin_amdgcn_perm(v_even_high, v_even_low, mask_even);
    uint32_t res_y = __builtin_amdgcn_perm(v_odd_high, v_odd_low, mask_odd);

    return make_int2(res_x, res_y);
}

// Method 2: LDS-based lookup (each thread has own copy in LDS)
// Note: This requires LDS table to be set up per workgroup
static __device__ __forceinline__ int2 lookup_lds(const uint32_t q4, const int8_t * __restrict__ lds_table) {
    int8_t results_even[4];
    int8_t results_odd[4];

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int byte = (q4 >> (i * 8)) & 0xff;
        results_even[i] = lds_table[byte & 0xf];
        results_odd[i]  = lds_table[byte >> 4];
    }

    uint32_t res_x = ((uint8_t)results_even[0]) |
                     ((uint8_t)results_even[1] << 8) |
                     ((uint8_t)results_even[2] << 16) |
                     ((uint8_t)results_even[3] << 24);
    uint32_t res_y = ((uint8_t)results_odd[0]) |
                     ((uint8_t)results_odd[1] << 8) |
                     ((uint8_t)results_odd[2] << 16) |
                     ((uint8_t)results_odd[3] << 24);

    return make_int2(res_x, res_y);
}

// Method 3: Simpler perm - reorganize table so we only need 2 perms
// Idea: Store table as two 8-byte tables, even indices in one, all in another
// Actually for MXFP4: values 0-7 are positive {0,1,2,3,4,6,8,12}
//                     values 8-15 are their negatives
// We can compute negative from positive: neg = -pos
static __device__ __forceinline__ int2 lookup_perm_simplified(const uint32_t q4) {
    // Positive values only: 0,1,2,3,4,6,8,12 - fits in 8 bytes!
    // Table: byte0=0, byte1=1, byte2=2, byte3=3, byte4=4, byte5=6, byte6=8, byte7=12
    const uint32_t pos_lo = 0x03020100;  // bytes 0-3: 0,1,2,3
    const uint32_t pos_hi = 0x0c080604;  // bytes 4-7: 4,6,8,12

    uint32_t q_even = (q4 & 0x0f0f0f0f);
    uint32_t q_odd  = (q4 >> 4) & 0x0f0f0f0f;

    // Get magnitude index (bits 0-2) and sign (bit 3)
    uint32_t mag_even = q_even & 0x07070707;
    uint32_t mag_odd  = q_odd  & 0x07070707;
    uint32_t sign_even = (q_even >> 3) & 0x01010101;
    uint32_t sign_odd  = (q_odd >> 3)  & 0x01010101;

    // Lookup positive values using perm (2 perms instead of 6)
    uint32_t val_even = __builtin_amdgcn_perm(pos_hi, pos_lo, mag_even);
    uint32_t val_odd  = __builtin_amdgcn_perm(pos_hi, pos_lo, mag_odd);

    // Apply sign: if sign bit set, negate
    // For each byte: result = sign ? -val : val = val ^ (sign * 0xff) - sign
    // Or: result = (val ^ neg_mask) + sign where neg_mask = sign * 0xff
    uint32_t neg_mask_even = sign_even * 0xff;
    uint32_t neg_mask_odd  = sign_odd  * 0xff;

    // Two's complement: -x = ~x + 1 = (x ^ 0xff) + 1
    // For signed byte: if sign, do (val ^ 0xff) + 1
    // Packed version: we need to handle carry per byte...
    // Simpler: use XOR and subtract
    // -val = ~val + 1, so: (sign ? -val : val) = (val ^ neg_mask) + sign
    // But this doesn't work for packed bytes due to carry

    // Use SIMD byte operations
    // Actually on AMD, we can use v_sub_u8 or pack/unpack
    // Let's try: result = val - 2 * val * sign = val * (1 - 2*sign)
    // That's still complex for packed bytes...

    // Simplest: unpack, negate, repack
    int8_t e0 = (int8_t)(val_even & 0xff);
    int8_t e1 = (int8_t)((val_even >> 8) & 0xff);
    int8_t e2 = (int8_t)((val_even >> 16) & 0xff);
    int8_t e3 = (int8_t)((val_even >> 24) & 0xff);

    if (sign_even & 0x01) e0 = -e0;
    if (sign_even & 0x0100) e1 = -e1;
    if (sign_even & 0x010000) e2 = -e2;
    if (sign_even & 0x01000000) e3 = -e3;

    int8_t o0 = (int8_t)(val_odd & 0xff);
    int8_t o1 = (int8_t)((val_odd >> 8) & 0xff);
    int8_t o2 = (int8_t)((val_odd >> 16) & 0xff);
    int8_t o3 = (int8_t)((val_odd >> 24) & 0xff);

    if (sign_odd & 0x01) o0 = -o0;
    if (sign_odd & 0x0100) o1 = -o1;
    if (sign_odd & 0x010000) o2 = -o2;
    if (sign_odd & 0x01000000) o3 = -o3;

    uint32_t res_x = ((uint8_t)e0) | ((uint8_t)e1 << 8) | ((uint8_t)e2 << 16) | ((uint8_t)e3 << 24);
    uint32_t res_y = ((uint8_t)o0) | ((uint8_t)o1 << 8) | ((uint8_t)o2 << 16) | ((uint8_t)o3 << 24);

    return make_int2(res_x, res_y);
}

// Method 4: Two perms + XOR trick for sign
// Key insight: the negative values are just the positive values with bit7 set
// Wait no, -1 = 0xff, -2 = 0xfe, etc. But 1 = 0x01, 2 = 0x02
// -x in two's complement = ~x + 1
// For small values: -1=0xff, -2=0xfe, -3=0xfd, -4=0xfc, -6=0xfa, -8=0xf8, -12=0xf4
// Can we compute this efficiently?
// -x = 256 - x for bytes, or (0 - x) & 0xff
static __device__ __forceinline__ int2 lookup_perm_v2(const uint32_t q4) {
    // Positive values: 0,1,2,3,4,6,8,12 as unsigned bytes
    const uint32_t pos_lo = 0x03020100;
    const uint32_t pos_hi = 0x0c080604;

    uint32_t q_even = (q4 & 0x0f0f0f0f);
    uint32_t q_odd  = (q4 >> 4) & 0x0f0f0f0f;

    uint32_t mag_even = q_even & 0x07070707;
    uint32_t mag_odd  = q_odd  & 0x07070707;

    // Lookup positive values
    uint32_t val_even = __builtin_amdgcn_perm(pos_hi, pos_lo, mag_even);
    uint32_t val_odd  = __builtin_amdgcn_perm(pos_hi, pos_lo, mag_odd);

    // Sign handling using packed byte subtraction
    // We want: if sign bit set, result = -val, else result = val
    // Trick: -val = 0 - val (saturated would be wrong, need wrapping)
    // Using v_sub: 0 - val gives -val

    // Create sign masks
    uint32_t sign_even = (q_even >> 3) & 0x01010101;
    uint32_t sign_odd  = (q_odd >> 3)  & 0x01010101;

    // Branchless: result = val - 2*val*sign = val*(1-2*sign)
    // Or use: result = (val ^ neg_mask) - neg_mask where neg_mask = -sign = (sign ? 0xff : 0)
    // -sign in byte = 0 - sign, which is 0 or 0xff
    uint32_t neg_mask_even = 0 - sign_even;  // 0x00 or 0xff per byte
    uint32_t neg_mask_odd  = 0 - sign_odd;

    // (val ^ neg_mask) - neg_mask:
    // if neg_mask=0: val ^ 0 - 0 = val
    // if neg_mask=0xff: val ^ 0xff - 0xff = ~val - 0xff = ~val + 1 = -val (two's complement!)
    // Wait: -0xff = +1 in byte arithmetic, so: ~val + 1 = -val. Yes!

    uint32_t res_x = (val_even ^ neg_mask_even) - neg_mask_even;
    uint32_t res_y = (val_odd ^ neg_mask_odd) - neg_mask_odd;

    return make_int2(res_x, res_y);
}

// Benchmark kernels
__global__ void bench_perm(uint32_t* input, int2* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int2 acc = make_int2(0, 0);
    uint32_t val = input[idx % n];

    for (int i = 0; i < ITERATIONS; i++) {
        int2 v = lookup_perm(val ^ i, kvalues_mxfp4_const);
        acc.x += v.x;
        acc.y += v.y;
        val = val ^ v.x;
    }
    output[idx] = acc;
}

__global__ void bench_lds(uint32_t* input, int2* output, int n) {
    __shared__ int8_t lds_table[16];

    // Initialize LDS table (one thread does it)
    if (threadIdx.x < 16) {
        lds_table[threadIdx.x] = kvalues_mxfp4_const[threadIdx.x];
    }
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int2 acc = make_int2(0, 0);
    uint32_t val = input[idx % n];

    for (int i = 0; i < ITERATIONS; i++) {
        int2 v = lookup_lds(val ^ i, lds_table);
        acc.x += v.x;
        acc.y += v.y;
        val = val ^ v.x;
    }
    output[idx] = acc;
}

__global__ void bench_perm_v2(uint32_t* input, int2* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int2 acc = make_int2(0, 0);
    uint32_t val = input[idx % n];

    for (int i = 0; i < ITERATIONS; i++) {
        int2 v = lookup_perm_v2(val ^ i);
        acc.x += v.x;
        acc.y += v.y;
        val = val ^ v.x;
    }
    output[idx] = acc;
}

// Correctness check
__global__ void test_correctness(int* errors) {
    int idx = threadIdx.x;
    if (idx >= 256) return;

    __shared__ int8_t lds_table[16];
    if (idx < 16) {
        lds_table[idx] = kvalues_mxfp4_const[idx];
    }
    __syncthreads();

    uint32_t test_val = idx | (idx << 8) | (idx << 16) | (idx << 24);

    int2 ref = lookup_perm(test_val, kvalues_mxfp4_const);
    int2 lds = lookup_lds(test_val, lds_table);
    int2 v2  = lookup_perm_v2(test_val);

    if (ref.x != lds.x || ref.y != lds.y) {
        atomicAdd(errors, 1);
        if (idx < 8) printf("LDS mismatch at %d: ref(%08x,%08x) lds(%08x,%08x)\n",
                           idx, ref.x, ref.y, lds.x, lds.y);
    }
    if (ref.x != v2.x || ref.y != v2.y) {
        atomicAdd(errors, 1);
        if (idx < 8) printf("V2 mismatch at %d: ref(%08x,%08x) v2(%08x,%08x)\n",
                           idx, ref.x, ref.y, v2.x, v2.y);
    }
}

int main() {
    printf("=== MXFP4 Lookup Methods Benchmark ===\n");
    printf("Iterations: %d, Threads: %d\n\n", ITERATIONS, THREADS * BLOCKS);

    uint32_t* d_input;
    int2* d_output;
    int* d_errors;

    hipMalloc(&d_input, THREADS * BLOCKS * sizeof(uint32_t));
    hipMalloc(&d_output, THREADS * BLOCKS * sizeof(int2));
    hipMalloc(&d_errors, sizeof(int));

    uint32_t* h_input = new uint32_t[THREADS * BLOCKS];
    for (int i = 0; i < THREADS * BLOCKS; i++) h_input[i] = i * 0x12345678;
    hipMemcpy(d_input, h_input, THREADS * BLOCKS * sizeof(uint32_t), hipMemcpyHostToDevice);

    // Correctness test
    printf("Testing correctness...\n");
    int h_errors = 0;
    hipMemcpy(d_errors, &h_errors, sizeof(int), hipMemcpyHostToDevice);
    test_correctness<<<1, 256>>>(d_errors);
    hipDeviceSynchronize();
    hipMemcpy(&h_errors, d_errors, sizeof(int), hipMemcpyDeviceToHost);

    if (h_errors > 0) {
        printf("FAILED: %d errors\n", h_errors);
        return 1;
    }
    printf("All methods correct!\n\n");

    // Warmup
    bench_perm<<<BLOCKS, THREADS>>>(d_input, d_output, THREADS * BLOCKS);
    bench_lds<<<BLOCKS, THREADS>>>(d_input, d_output, THREADS * BLOCKS);
    bench_perm_v2<<<BLOCKS, THREADS>>>(d_input, d_output, THREADS * BLOCKS);
    hipDeviceSynchronize();

    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);
    float ms;

    // Benchmark perm (original)
    hipEventRecord(start);
    bench_perm<<<BLOCKS, THREADS>>>(d_input, d_output, THREADS * BLOCKS);
    hipEventRecord(stop);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&ms, start, stop);
    float perm_ms = ms;
    printf("1. Perm (6 ops):    %7.2f ms\n", ms);

    // Benchmark LDS
    hipEventRecord(start);
    bench_lds<<<BLOCKS, THREADS>>>(d_input, d_output, THREADS * BLOCKS);
    hipEventRecord(stop);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&ms, start, stop);
    printf("2. LDS lookup:      %7.2f ms (%.2fx)\n", ms, perm_ms/ms);

    // Benchmark perm v2 (2 perms + XOR)
    hipEventRecord(start);
    bench_perm_v2<<<BLOCKS, THREADS>>>(d_input, d_output, THREADS * BLOCKS);
    hipEventRecord(stop);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&ms, start, stop);
    printf("3. Perm v2 (2+XOR): %7.2f ms (%.2fx)\n", ms, perm_ms/ms);

    hipFree(d_input);
    hipFree(d_output);
    hipFree(d_errors);
    delete[] h_input;

    return 0;
}
