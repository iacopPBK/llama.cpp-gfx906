// MXFP4 Optimization using SDWA for byte-level operations
// SDWA (Sub-Dword Addressing) allows per-byte operations on GFX906
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
#include <chrono>

//=============================================================================
// Reference: OOB zeroing (known correct)
//=============================================================================
__device__ __forceinline__
void ref_oob(uint32_t q, int8_t* result_even, int8_t* result_odd) {
    constexpr uint32_t pos_tbl = 0x0C080604u;
    constexpr uint32_t pos_tbl2 = 0x03020100u;
    constexpr uint32_t neg_tbl = 0xF4F8FAFCu;
    constexpr uint32_t neg_tbl2 = 0xFDFEFF00u;

    uint32_t q_even = q & 0x0F0F0F0Fu;
    uint32_t q_odd  = (q >> 4) & 0x0F0F0F0Fu;

    uint32_t pos_idx_even = (q_even & 0x07070707u) | ((q_even & 0x08080808u) << 4);
    uint32_t pos_idx_odd  = (q_odd  & 0x07070707u) | ((q_odd  & 0x08080808u) << 4);

    uint32_t neg_idx_even = ((q_even ^ 0x08080808u) & 0x07070707u) | (((q_even ^ 0x08080808u) & 0x08080808u) << 4);
    uint32_t neg_idx_odd  = ((q_odd  ^ 0x08080808u) & 0x07070707u) | (((q_odd  ^ 0x08080808u) & 0x08080808u) << 4);

    uint32_t pos_even, pos_odd, neg_even, neg_odd;
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(pos_even) : "v"(pos_tbl), "v"(pos_tbl2), "v"(pos_idx_even));
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(neg_even) : "v"(neg_tbl), "v"(neg_tbl2), "v"(neg_idx_even));
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(pos_odd)  : "v"(pos_tbl), "v"(pos_tbl2), "v"(pos_idx_odd));
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(neg_odd)  : "v"(neg_tbl), "v"(neg_tbl2), "v"(neg_idx_odd));

    *reinterpret_cast<uint32_t*>(result_even) = pos_even | neg_even;
    *reinterpret_cast<uint32_t*>(result_odd)  = pos_odd  | neg_odd;
}

//=============================================================================
// Optimization H: 3-perm approach using magnitude lookup + sign selection
// Use single magnitude lookup, then use perm to expand sign and BFI to select
//=============================================================================
__device__ __forceinline__
void opt_h_3perm(uint32_t q, int8_t* result_even, int8_t* result_odd) {
    // Table: positive magnitude {0,1,2,3,4,6,8,12} and negative {0,-1,-2,-3,-4,-6,-8,-12}
    // Pack both in one lookup by using different byte positions
    constexpr uint32_t mag_lo = 0x03020100u;   // {0,1,2,3}
    constexpr uint32_t mag_hi = 0x0C080604u;   // {4,6,8,12}
    constexpr uint32_t neg_lo = 0xFDFEFF00u;   // {0,-1,-2,-3}
    constexpr uint32_t neg_hi = 0xF4F8FAFCu;   // {-4,-6,-8,-12}

    uint32_t q_even = q & 0x0F0F0F0Fu;
    uint32_t q_odd  = (q >> 4) & 0x0F0F0F0Fu;
    uint32_t idx_even = q_even & 0x07070707u;
    uint32_t idx_odd  = q_odd & 0x07070707u;

    // Lookup magnitude (positive values)
    uint32_t mag_even, mag_odd;
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(mag_even) : "v"(mag_hi), "v"(mag_lo), "v"(idx_even));
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(mag_odd)  : "v"(mag_hi), "v"(mag_lo), "v"(idx_odd));

    // Lookup negative values
    uint32_t neg_even, neg_odd;
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(neg_even) : "v"(neg_hi), "v"(neg_lo), "v"(idx_even));
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(neg_odd)  : "v"(neg_hi), "v"(neg_lo), "v"(idx_odd));

    // Select based on sign bit using perm (3rd perm - merge)
    // If bit3=0: select from bytes 0-3 (positive), if bit3=1: select from bytes 4-7 (negative)
    // merge_idx = byte_pos + (sign_bit ? 4 : 0) = byte_pos | (sign >> 1)
    uint32_t merge_even = 0x03020100u | ((q_even & 0x08080808u) >> 1);
    uint32_t merge_odd  = 0x03020100u | ((q_odd  & 0x08080808u) >> 1);

    uint32_t res_even, res_odd;
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(res_even) : "v"(neg_even), "v"(mag_even), "v"(merge_even));
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(res_odd)  : "v"(neg_odd),  "v"(mag_odd),  "v"(merge_odd));

    *reinterpret_cast<uint32_t*>(result_even) = res_even;
    *reinterpret_cast<uint32_t*>(result_odd)  = res_odd;
}

//=============================================================================
// Optimization I: Ultra-minimal - can we get to 4 perms total?
// Process even and odd together with shared operations
//=============================================================================
__device__ __forceinline__
void opt_i_4perm_total(uint32_t q, int8_t* result_even, int8_t* result_odd) {
    // INSIGHT: We can save perms by computing both even and odd results
    // from a single set of table lookups if we're clever about indexing.
    //
    // Current: 6 perms = 2 (mag) + 2 (neg) + 2 (merge) per even/odd
    // With OOB: 4 perms = 2 + 2 for even/odd (no merge needed)
    //
    // Can we do 2 perms total for mag (shared) + 2 for merge?
    // The issue is that even nibbles are at positions 0,8,16,24
    // and odd nibbles are at positions 4,12,20,28 in the original q.

    // Attempt: Process interleaved nibbles
    // Actually the fundamental constraint is that v_perm processes 4 bytes
    // and we have 8 nibbles per 32-bit word, requiring 8 byte outputs.
    // Minimum is 2 v_perm calls for 8 bytes.

    // Let's try the OOB approach but with slightly optimized index calculation
    constexpr uint32_t pos_tbl = 0x0C080604u;
    constexpr uint32_t pos_tbl2 = 0x03020100u;
    constexpr uint32_t neg_tbl = 0xF4F8FAFCu;
    constexpr uint32_t neg_tbl2 = 0xFDFEFF00u;

    // Extract nibbles using combined operations
    uint32_t q_even = q & 0x0F0F0F0Fu;
    uint32_t q_odd  = (q >> 4) & 0x0F0F0F0Fu;

    // Build pos index: (idx & 7) | ((idx & 8) << 4)
    // Use v_lshl_add if available, or v_and_or
    uint32_t idx7_e = q_even & 0x07070707u;
    uint32_t idx7_o = q_odd  & 0x07070707u;
    uint32_t bit3_e = (q_even & 0x08080808u) << 4;
    uint32_t bit3_o = (q_odd  & 0x08080808u) << 4;

    uint32_t pos_idx_e = idx7_e | bit3_e;
    uint32_t pos_idx_o = idx7_o | bit3_o;
    uint32_t neg_idx_e = idx7_e | (bit3_e ^ 0x80808080u);
    uint32_t neg_idx_o = idx7_o | (bit3_o ^ 0x80808080u);

    // 4 perm lookups
    uint32_t pos_e, pos_o, neg_e, neg_o;
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(pos_e) : "v"(pos_tbl), "v"(pos_tbl2), "v"(pos_idx_e));
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(neg_e) : "v"(neg_tbl), "v"(neg_tbl2), "v"(neg_idx_e));
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(pos_o) : "v"(pos_tbl), "v"(pos_tbl2), "v"(pos_idx_o));
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(neg_o) : "v"(neg_tbl), "v"(neg_tbl2), "v"(neg_idx_o));

    *reinterpret_cast<uint32_t*>(result_even) = pos_e | neg_e;
    *reinterpret_cast<uint32_t*>(result_odd)  = pos_o | neg_o;
}

//=============================================================================
// Optimization J: Shared nibble extraction with reduced masking
//=============================================================================
__device__ __forceinline__
void opt_j_shared_nibble(uint32_t q, int8_t* result_even, int8_t* result_odd) {
    constexpr uint32_t pos_tbl = 0x0C080604u;
    constexpr uint32_t pos_tbl2 = 0x03020100u;
    constexpr uint32_t neg_tbl = 0xF4F8FAFCu;
    constexpr uint32_t neg_tbl2 = 0xFDFEFF00u;

    // Pre-compute common sub-expressions
    uint32_t q_masked = q & 0x8F8F8F8Fu;  // Keep bits 3,2,1,0,7 of each byte
    uint32_t q_even = q_masked & 0x0F0F0F0Fu;  // Just lower nibble
    uint32_t q_odd  = (q >> 4) & 0x0F0F0F0Fu;  // Extract upper nibble

    // Common: idx7 = lower 3 bits
    uint32_t idx7_e = q_even & 0x07070707u;
    uint32_t idx7_o = q_odd  & 0x07070707u;

    // Sign bits already in position for OOB calculation
    uint32_t sign_e = q_even & 0x08080808u;
    uint32_t sign_o = q_odd  & 0x08080808u;

    // Shift sign to bit 7 position
    uint32_t oob_e = sign_e << 4;
    uint32_t oob_o = sign_o << 4;

    // Build indices
    uint32_t pos_idx_e = idx7_e | oob_e;
    uint32_t pos_idx_o = idx7_o | oob_o;
    uint32_t neg_idx_e = idx7_e | (oob_e ^ 0x80808080u);
    uint32_t neg_idx_o = idx7_o | (oob_o ^ 0x80808080u);

    uint32_t pos_e, pos_o, neg_e, neg_o;
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(pos_e) : "v"(pos_tbl), "v"(pos_tbl2), "v"(pos_idx_e));
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(neg_e) : "v"(neg_tbl), "v"(neg_tbl2), "v"(neg_idx_e));
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(pos_o) : "v"(pos_tbl), "v"(pos_tbl2), "v"(pos_idx_o));
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(neg_o) : "v"(neg_tbl), "v"(neg_tbl2), "v"(neg_idx_o));

    *reinterpret_cast<uint32_t*>(result_even) = pos_e | neg_e;
    *reinterpret_cast<uint32_t*>(result_odd)  = pos_o | neg_o;
}

//=============================================================================
// Optimization K: Try v_mad_u32_u24 for index computation
// v_mad_u32_u24 D, S0, S1, S2 : D = S0 * S1 + S2 (24-bit multiply, useful for scaling)
//=============================================================================
__device__ __forceinline__
void opt_k_mad(uint32_t q, int8_t* result_even, int8_t* result_odd) {
    constexpr uint32_t pos_tbl = 0x0C080604u;
    constexpr uint32_t pos_tbl2 = 0x03020100u;
    constexpr uint32_t neg_tbl = 0xF4F8FAFCu;
    constexpr uint32_t neg_tbl2 = 0xFDFEFF00u;

    uint32_t q_even = q & 0x0F0F0F0Fu;
    uint32_t q_odd  = (q >> 4) & 0x0F0F0F0Fu;

    // pos_idx = (idx & 7) | ((idx >> 3) << 7)
    //         = (idx & 7) | ((idx & 8) << 4)
    // Combine using: pos_idx = (idx & 7) + ((idx >> 3) * 0x10101010)
    // But we need to shift the multiplication result... doesn't help

    // Stick with the fast path
    uint32_t idx7_e = q_even & 0x07070707u;
    uint32_t idx7_o = q_odd  & 0x07070707u;
    uint32_t oob_e = (q_even << 4) & 0x80808080u;
    uint32_t oob_o = (q_odd << 4) & 0x80808080u;

    uint32_t pos_idx_e = idx7_e | oob_e;
    uint32_t pos_idx_o = idx7_o | oob_o;
    uint32_t neg_idx_e = idx7_e | (oob_e ^ 0x80808080u);
    uint32_t neg_idx_o = idx7_o | (oob_o ^ 0x80808080u);

    uint32_t pos_e, pos_o, neg_e, neg_o;
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(pos_e) : "v"(pos_tbl), "v"(pos_tbl2), "v"(pos_idx_e));
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(neg_e) : "v"(neg_tbl), "v"(neg_tbl2), "v"(neg_idx_e));
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(pos_o) : "v"(pos_tbl), "v"(pos_tbl2), "v"(pos_idx_o));
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(neg_o) : "v"(neg_tbl), "v"(neg_tbl2), "v"(neg_idx_o));

    *reinterpret_cast<uint32_t*>(result_even) = pos_e | neg_e;
    *reinterpret_cast<uint32_t*>(result_odd)  = pos_o | neg_o;
}

//=============================================================================
// Optimization L: Use v_add_lshl_u32 or v_lshl_add_u32 for combined ops
// GFX9 has v_lshl_add_u32 D, S0, S1, S2 : D = (S0 << S1) + S2
//=============================================================================
__device__ __forceinline__
void opt_l_lshl_add(uint32_t q, int8_t* result_even, int8_t* result_odd) {
    constexpr uint32_t pos_tbl = 0x0C080604u;
    constexpr uint32_t pos_tbl2 = 0x03020100u;
    constexpr uint32_t neg_tbl = 0xF4F8FAFCu;
    constexpr uint32_t neg_tbl2 = 0xFDFEFF00u;

    uint32_t q_even = q & 0x0F0F0F0Fu;
    uint32_t q_odd  = (q >> 4) & 0x0F0F0F0Fu;

    // idx7
    uint32_t idx7_e = q_even & 0x07070707u;
    uint32_t idx7_o = q_odd  & 0x07070707u;

    // sign bits
    uint32_t sign_e = (q_even & 0x08080808u);
    uint32_t sign_o = (q_odd  & 0x08080808u);

    // Use v_lshl_add: (sign << 4) + idx7 = lshl_add(sign, 4, idx7)
    uint32_t pos_idx_e, pos_idx_o;
    asm volatile("v_lshl_add_u32 %0, %1, 4, %2" : "=v"(pos_idx_e) : "v"(sign_e), "v"(idx7_e));
    asm volatile("v_lshl_add_u32 %0, %1, 4, %2" : "=v"(pos_idx_o) : "v"(sign_o), "v"(idx7_o));

    // For neg: XOR the shifted sign bit
    uint32_t neg_idx_e = pos_idx_e ^ 0x80808080u;
    uint32_t neg_idx_o = pos_idx_o ^ 0x80808080u;

    uint32_t pos_e, pos_o, neg_e, neg_o;
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(pos_e) : "v"(pos_tbl), "v"(pos_tbl2), "v"(pos_idx_e));
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(neg_e) : "v"(neg_tbl), "v"(neg_tbl2), "v"(neg_idx_e));
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(pos_o) : "v"(pos_tbl), "v"(pos_tbl2), "v"(pos_idx_o));
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(neg_o) : "v"(neg_tbl), "v"(neg_tbl2), "v"(neg_idx_o));

    *reinterpret_cast<uint32_t*>(result_even) = pos_e | neg_e;
    *reinterpret_cast<uint32_t*>(result_odd)  = pos_o | neg_o;
}

//=============================================================================
// Benchmarks
//=============================================================================

__global__ void bench_ref(uint32_t* __restrict__ out, const uint32_t* __restrict__ in, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    uint32_t q = in[idx];
    int8_t re[4], ro[4];
    #pragma unroll
    for (int i = 0; i < 100; i++) {
        ref_oob(q, re, ro);
        q ^= *reinterpret_cast<uint32_t*>(re) ^ *reinterpret_cast<uint32_t*>(ro);
    }
    out[idx*2] = *reinterpret_cast<uint32_t*>(re);
    out[idx*2+1] = *reinterpret_cast<uint32_t*>(ro);
}

__global__ void bench_h(uint32_t* __restrict__ out, const uint32_t* __restrict__ in, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    uint32_t q = in[idx];
    int8_t re[4], ro[4];
    #pragma unroll
    for (int i = 0; i < 100; i++) {
        opt_h_3perm(q, re, ro);
        q ^= *reinterpret_cast<uint32_t*>(re) ^ *reinterpret_cast<uint32_t*>(ro);
    }
    out[idx*2] = *reinterpret_cast<uint32_t*>(re);
    out[idx*2+1] = *reinterpret_cast<uint32_t*>(ro);
}

__global__ void bench_i(uint32_t* __restrict__ out, const uint32_t* __restrict__ in, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    uint32_t q = in[idx];
    int8_t re[4], ro[4];
    #pragma unroll
    for (int i = 0; i < 100; i++) {
        opt_i_4perm_total(q, re, ro);
        q ^= *reinterpret_cast<uint32_t*>(re) ^ *reinterpret_cast<uint32_t*>(ro);
    }
    out[idx*2] = *reinterpret_cast<uint32_t*>(re);
    out[idx*2+1] = *reinterpret_cast<uint32_t*>(ro);
}

__global__ void bench_l(uint32_t* __restrict__ out, const uint32_t* __restrict__ in, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    uint32_t q = in[idx];
    int8_t re[4], ro[4];
    #pragma unroll
    for (int i = 0; i < 100; i++) {
        opt_l_lshl_add(q, re, ro);
        q ^= *reinterpret_cast<uint32_t*>(re) ^ *reinterpret_cast<uint32_t*>(ro);
    }
    out[idx*2] = *reinterpret_cast<uint32_t*>(re);
    out[idx*2+1] = *reinterpret_cast<uint32_t*>(ro);
}

//=============================================================================
// Verification
//=============================================================================

__global__ void verify(int* errors) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 65536) return;
    uint32_t q = idx | (idx << 16);
    int8_t ref_e[4], ref_o[4];
    int8_t h_e[4], h_o[4];
    int8_t i_e[4], i_o[4];
    int8_t l_e[4], l_o[4];

    ref_oob(q, ref_e, ref_o);
    opt_h_3perm(q, h_e, h_o);
    opt_i_4perm_total(q, i_e, i_o);
    opt_l_lshl_add(q, l_e, l_o);

    uint32_t rv_e = *reinterpret_cast<uint32_t*>(ref_e);
    uint32_t rv_o = *reinterpret_cast<uint32_t*>(ref_o);

    if (*reinterpret_cast<uint32_t*>(h_e) != rv_e || *reinterpret_cast<uint32_t*>(h_o) != rv_o)
        atomicAdd(&errors[0], 1);
    if (*reinterpret_cast<uint32_t*>(i_e) != rv_e || *reinterpret_cast<uint32_t*>(i_o) != rv_o)
        atomicAdd(&errors[1], 1);
    if (*reinterpret_cast<uint32_t*>(l_e) != rv_e || *reinterpret_cast<uint32_t*>(l_o) != rv_o)
        atomicAdd(&errors[2], 1);
}

int main() {
    printf("MXFP4 SDWA/Fused Operation Tests\n");
    printf("================================\n\n");

    // Verify
    int* d_errors;
    hipMalloc(&d_errors, 4 * sizeof(int));
    hipMemset(d_errors, 0, 4 * sizeof(int));
    verify<<<256, 256>>>(d_errors);
    hipDeviceSynchronize();

    int h_errors[4];
    hipMemcpy(h_errors, d_errors, 4 * sizeof(int), hipMemcpyDeviceToHost);
    printf("Opt H (3-perm merge) errors: %d\n", h_errors[0]);
    printf("Opt I (4-perm total) errors: %d\n", h_errors[1]);
    printf("Opt L (lshl_add)     errors: %d\n", h_errors[2]);
    hipFree(d_errors);

    if (h_errors[0] || h_errors[1] || h_errors[2]) {
        printf("\n⚠️  Some have errors!\n\n");
    } else {
        printf("\n✓ All correct!\n\n");
    }

    // Benchmark
    const int N = 1024 * 1024;
    uint32_t *d_in, *d_out;
    hipMalloc(&d_in, N * sizeof(uint32_t));
    hipMalloc(&d_out, N * 2 * sizeof(uint32_t));

    uint32_t* h_in = new uint32_t[N];
    for (int i = 0; i < N; i++) h_in[i] = rand();
    hipMemcpy(d_in, h_in, N * sizeof(uint32_t), hipMemcpyHostToDevice);
    delete[] h_in;

    int bs = 256, gs = (N + bs - 1) / bs;

    auto bench = [&](const char* name, auto kernel) {
        for (int i = 0; i < 5; i++) kernel<<<gs, bs>>>(d_out, d_in, N);
        hipDeviceSynchronize();

        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 20; i++) kernel<<<gs, bs>>>(d_out, d_in, N);
        hipDeviceSynchronize();
        auto t1 = std::chrono::high_resolution_clock::now();

        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / 20;
        double gops = (N * 100.0) / (ms / 1000.0) / 1e9;
        printf("%-22s: %.3f ms, %.2f Gops/s\n", name, ms, gops);
    };

    printf("Benchmark Results:\n");
    bench("Ref (OOB 4-perm)", bench_ref);
    bench("Opt H (6-perm merge)", bench_h);
    bench("Opt I (4-perm OOB)", bench_i);
    bench("Opt L (lshl_add)", bench_l);

    hipFree(d_in);
    hipFree(d_out);
    return 0;
}
