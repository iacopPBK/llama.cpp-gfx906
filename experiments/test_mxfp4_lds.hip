// Test MXFP4 lookup using LDS instead of v_perm
// The idea: put the 16-byte table in LDS, use ds_read_u8 to fetch

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>

__device__ __constant__ int8_t kvalues_mxfp4[16] = {
    0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12
};

// ============================================================================
// REFERENCE: Original 6 v_perm
// ============================================================================
__device__ __forceinline__ int2 lookup_reference(uint32_t q4) {
    const uint32_t *values = (const uint32_t *)kvalues_mxfp4;
    const uint32_t q_even = q4;
    const uint32_t q_odd  = (q4 >> 4);

    uint32_t v_even_low  = __builtin_amdgcn_perm(values[1], values[0], q_even & 0x07070707);
    uint32_t v_odd_low   = __builtin_amdgcn_perm(values[1], values[0], q_odd  & 0x07070707);
    uint32_t v_even_high = __builtin_amdgcn_perm(values[3], values[2], q_even & 0x07070707);
    uint32_t v_odd_high  = __builtin_amdgcn_perm(values[3], values[2], q_odd  & 0x07070707);

    uint32_t mask_even = 0x03020100 | ((q_even & 0x08080808) >> 1);
    uint32_t mask_odd  = 0x03020100 | ((q_odd  & 0x08080808) >> 1);
    uint32_t res_x = __builtin_amdgcn_perm(v_even_high, v_even_low, mask_even);
    uint32_t res_y = __builtin_amdgcn_perm(v_odd_high,  v_odd_low,  mask_odd);

    return make_int2(res_x, res_y);
}

// ============================================================================
// LDS-based lookup: Load table into LDS, use ds_read for lookup
// ============================================================================
__shared__ int8_t lds_table[16];

__device__ __forceinline__ int2 lookup_lds(uint32_t q4) {
    // Extract 8 nibble indices
    uint32_t idx0 = (q4 >>  0) & 0xF;
    uint32_t idx1 = (q4 >>  4) & 0xF;
    uint32_t idx2 = (q4 >>  8) & 0xF;
    uint32_t idx3 = (q4 >> 12) & 0xF;
    uint32_t idx4 = (q4 >> 16) & 0xF;
    uint32_t idx5 = (q4 >> 20) & 0xF;
    uint32_t idx6 = (q4 >> 24) & 0xF;
    uint32_t idx7 = (q4 >> 28) & 0xF;

    // LDS reads (using pointer arithmetic instead of inline asm for simplicity)
    int8_t v0 = lds_table[idx0];
    int8_t v1 = lds_table[idx1];
    int8_t v2 = lds_table[idx2];
    int8_t v3 = lds_table[idx3];
    int8_t v4 = lds_table[idx4];
    int8_t v5 = lds_table[idx5];
    int8_t v6 = lds_table[idx6];
    int8_t v7 = lds_table[idx7];

    // Pack into int2
    uint32_t res_x = ((uint8_t)v0) | (((uint8_t)v2) << 8) | (((uint8_t)v4) << 16) | (((uint8_t)v6) << 24);
    uint32_t res_y = ((uint8_t)v1) | (((uint8_t)v3) << 8) | (((uint8_t)v5) << 16) | (((uint8_t)v7) << 24);

    return make_int2(res_x, res_y);
}

// ============================================================================
// LDS with ds_read_u8 inline assembly
// ============================================================================
__device__ __forceinline__ int2 lookup_lds_asm(uint32_t q4, uint32_t lds_base) {
    uint32_t v0, v1, v2, v3, v4, v5, v6, v7;

    // Extract nibble indices and compute LDS offsets
    uint32_t off0 = lds_base + ((q4 >>  0) & 0xF);
    uint32_t off1 = lds_base + ((q4 >>  4) & 0xF);
    uint32_t off2 = lds_base + ((q4 >>  8) & 0xF);
    uint32_t off3 = lds_base + ((q4 >> 12) & 0xF);
    uint32_t off4 = lds_base + ((q4 >> 16) & 0xF);
    uint32_t off5 = lds_base + ((q4 >> 20) & 0xF);
    uint32_t off6 = lds_base + ((q4 >> 24) & 0xF);
    uint32_t off7 = lds_base + ((q4 >> 28) & 0xF);

    // ds_read_u8: loads a single byte from LDS with sign extension
    asm volatile("ds_read_u8 %0, %1" : "=v"(v0) : "v"(off0));
    asm volatile("ds_read_u8 %0, %1" : "=v"(v1) : "v"(off1));
    asm volatile("ds_read_u8 %0, %1" : "=v"(v2) : "v"(off2));
    asm volatile("ds_read_u8 %0, %1" : "=v"(v3) : "v"(off3));
    asm volatile("ds_read_u8 %0, %1" : "=v"(v4) : "v"(off4));
    asm volatile("ds_read_u8 %0, %1" : "=v"(v5) : "v"(off5));
    asm volatile("ds_read_u8 %0, %1" : "=v"(v6) : "v"(off6));
    asm volatile("ds_read_u8 %0, %1" : "=v"(v7) : "v"(off7));

    // Wait for LDS reads
    asm volatile("s_waitcnt lgkmcnt(0)");

    // Pack results (even nibbles to x, odd nibbles to y)
    uint32_t res_x = (v0 & 0xFF) | ((v2 & 0xFF) << 8) | ((v4 & 0xFF) << 16) | ((v6 & 0xFF) << 24);
    uint32_t res_y = (v1 & 0xFF) | ((v3 & 0xFF) << 8) | ((v5 & 0xFF) << 16) | ((v7 & 0xFF) << 24);

    return make_int2(res_x, res_y);
}

// ============================================================================
// Minimal v_perm: 3 v_perm using interleaved table + direct indexing
// Idea: Store pos/neg interleaved, use single lookup + v_perm for assembly
// ============================================================================
__device__ __forceinline__ int2 lookup_3perm_attempt(uint32_t q4) {
    // This is an attempt at a 3 v_perm solution
    // Interleaved table: {pos[0], neg[0], pos[1], neg[1], ...}
    // Layout: {0, 0, 1, -1, 2, -2, 3, -3, 4, -4, 6, -6, 8, -8, 12, -12}
    // Index mapping: new_idx = (magnitude << 1) | sign

    // But this still requires 16-entry table access which needs 2 v_perm + selection

    // Actually, let's try something different:
    // What if we use 3 v_perm in a creative way?
    // v_perm 1: Look up values for nibbles 0-3 (from even positions)
    // v_perm 2: Look up values for nibbles 4-7 (from odd positions)
    // v_perm 3: Combine/rearrange

    // Wait, this doesn't work because v_perm can only access 8 entries at a time

    // Let me try: Use the ORIGINAL table layout
    // {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12}
    //
    // Instead of looking up each nibble, what if we process differently?
    //
    // Observation: For a 32-bit q4 with 8 nibbles, we get 8 output bytes.
    // The original approach processes: even nibbles -> res_x, odd nibbles -> res_y
    //
    // What if we could do a single pass that handles both?

    // The indices for even nibbles are bits [3:0], [11:8], [19:16], [27:24] of q4
    // The indices for odd nibbles are bits [7:4], [15:12], [23:20], [31:28] of q4

    // Using the magnitude-only table:
    const uint32_t mag_lo = 0x03020100;  // {0, 1, 2, 3}
    const uint32_t mag_hi = 0x0c080604;  // {4, 6, 8, 12}
    const uint32_t neg_lo = 0xfdfeff00;  // {0, -1, -2, -3}
    const uint32_t neg_hi = 0xf4f8fafc;  // {-4, -6, -8, -12}

    // v_perm 1: magnitude for even nibbles
    uint32_t q_even = q4;
    uint32_t q_odd = q4 >> 4;
    uint32_t idx_even = q_even & 0x07070707;
    uint32_t idx_odd = q_odd & 0x07070707;

    // We need 4 lookups total (2 for pos, 2 for neg) or
    // 2 lookups (pos) + arithmetic negation... but we showed that's slower

    // The problem is: 16 entries > 8 bytes accessible per v_perm
    // There's no way around this with v_perm alone

    // Let's just fall back to reference
    return lookup_reference(q4);
}

// ============================================================================
// Test kernels
// ============================================================================
__global__ void test_correctness(uint32_t* inputs, int2* ref, int2* lds_res, int n) {
    int tid = threadIdx.x;

    // Initialize LDS table (only thread 0)
    if (tid < 16) {
        lds_table[tid] = kvalues_mxfp4[tid];
    }
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    uint32_t q4 = inputs[idx];
    ref[idx] = lookup_reference(q4);
    lds_res[idx] = lookup_lds(q4);
}

__global__ void bench_reference(uint32_t* inputs, int2* outputs, int n, int iters) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    uint32_t q4 = inputs[idx];
    int2 result = make_int2(0, 0);
    #pragma unroll 1
    for (int i = 0; i < iters; i++) {
        int2 r = lookup_reference(q4 ^ i);
        result.x += r.x; result.y += r.y;
    }
    outputs[idx] = result;
}

__global__ void bench_lds(uint32_t* inputs, int2* outputs, int n, int iters) {
    int tid = threadIdx.x;
    if (tid < 16) lds_table[tid] = kvalues_mxfp4[tid];
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    uint32_t q4 = inputs[idx];
    int2 result = make_int2(0, 0);
    #pragma unroll 1
    for (int i = 0; i < iters; i++) {
        int2 r = lookup_lds(q4 ^ i);
        result.x += r.x; result.y += r.y;
    }
    outputs[idx] = result;
}

int main() {
    printf("=== MXFP4 LDS Lookup Test ===\n\n");

    const int N = 65536;
    const int ITERS = 10000;
    const int BLOCK = 256;
    const int GRID = (N + BLOCK - 1) / BLOCK;

    uint32_t* h_inputs = new uint32_t[N];
    int2* h_ref = new int2[N];
    int2* h_lds = new int2[N];

    for (int i = 0; i < N; i++) h_inputs[i] = i;

    uint32_t* d_inputs;
    int2 *d_ref, *d_lds;
    hipMalloc(&d_inputs, N * sizeof(uint32_t));
    hipMalloc(&d_ref, N * sizeof(int2));
    hipMalloc(&d_lds, N * sizeof(int2));
    hipMemcpy(d_inputs, h_inputs, N * sizeof(uint32_t), hipMemcpyHostToDevice);

    // Correctness test
    hipLaunchKernelGGL(test_correctness, dim3(GRID), dim3(BLOCK), 0, 0,
                       d_inputs, d_ref, d_lds, N);
    hipDeviceSynchronize();

    hipMemcpy(h_ref, d_ref, N * sizeof(int2), hipMemcpyDeviceToHost);
    hipMemcpy(h_lds, d_lds, N * sizeof(int2), hipMemcpyDeviceToHost);

    printf("=== Correctness ===\n");
    int errors = 0;
    for (int i = 0; i < N && errors < 3; i++) {
        if (h_ref[i].x != h_lds[i].x || h_ref[i].y != h_lds[i].y) {
            if (errors == 0) printf("[LDS] MISMATCH:\n");
            printf("  i=%d (0x%04x): ref=(0x%08x,0x%08x) got=(0x%08x,0x%08x)\n",
                   i, h_inputs[i], h_ref[i].x, h_ref[i].y, h_lds[i].x, h_lds[i].y);
            errors++;
        }
    }
    if (errors == 0) printf("[LDS lookup] PASSED\n");
    else printf("[LDS lookup] FAILED (%d+ errors)\n", errors);

    // Benchmark
    printf("\n=== Benchmark (%d x %d iters) ===\n", N, ITERS);

    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);

    // Warmup
    hipLaunchKernelGGL(bench_reference, dim3(GRID), dim3(BLOCK), 0, 0, d_inputs, d_ref, N, 100);
    hipDeviceSynchronize();

    // Reference
    hipEventRecord(start);
    hipLaunchKernelGGL(bench_reference, dim3(GRID), dim3(BLOCK), 0, 0, d_inputs, d_ref, N, ITERS);
    hipEventRecord(stop);
    hipEventSynchronize(stop);
    float ms_ref;
    hipEventElapsedTime(&ms_ref, start, stop);
    printf("Reference (6 v_perm): %.2f ms\n", ms_ref);

    // LDS
    hipEventRecord(start);
    hipLaunchKernelGGL(bench_lds, dim3(GRID), dim3(BLOCK), 0, 0, d_inputs, d_lds, N, ITERS);
    hipEventRecord(stop);
    hipEventSynchronize(stop);
    float ms_lds;
    hipEventElapsedTime(&ms_lds, start, stop);
    printf("LDS lookup:           %.2f ms (%.2fx)\n", ms_lds, ms_ref / ms_lds);

    // Cleanup
    hipFree(d_inputs); hipFree(d_ref); hipFree(d_lds);
    delete[] h_inputs; delete[] h_ref; delete[] h_lds;
    hipEventDestroy(start); hipEventDestroy(stop);

    printf("\n=== Done ===\n");
    return 0;
}
