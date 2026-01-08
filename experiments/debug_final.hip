// Debug the final implementations
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>

__device__ __forceinline__
void baseline_6perm(uint32_t q, int8_t* result_even, int8_t* result_odd) {
    constexpr uint32_t tbl_lo_pos = 0x03020100u;
    constexpr uint32_t tbl_hi_pos = 0x0C080604u;
    constexpr uint32_t tbl_lo_neg = 0xFDFEFF00u;
    constexpr uint32_t tbl_hi_neg = 0xF4F8FAFCu;

    uint32_t q_even = q & 0x0F0F0F0Fu;
    uint32_t q_odd  = (q >> 4) & 0x0F0F0F0Fu;
    uint32_t idx_even = q_even & 0x07070707u;
    uint32_t idx_odd  = q_odd & 0x07070707u;

    uint32_t v_e_lo, v_e_hi, v_e_merged;
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(v_e_lo) : "v"(tbl_hi_pos), "v"(tbl_lo_pos), "v"(idx_even));
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(v_e_hi) : "v"(tbl_hi_neg), "v"(tbl_lo_neg), "v"(idx_even));
    uint32_t mask_e = 0x03020100u | ((q_even & 0x08080808u) >> 1);
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(v_e_merged) : "v"(v_e_hi), "v"(v_e_lo), "v"(mask_e));

    uint32_t v_o_lo, v_o_hi, v_o_merged;
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(v_o_lo) : "v"(tbl_hi_pos), "v"(tbl_lo_pos), "v"(idx_odd));
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(v_o_hi) : "v"(tbl_hi_neg), "v"(tbl_lo_neg), "v"(idx_odd));
    uint32_t mask_o = 0x03020100u | ((q_odd & 0x08080808u) >> 1);
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(v_o_merged) : "v"(v_o_hi), "v"(v_o_lo), "v"(mask_o));

    *reinterpret_cast<uint32_t*>(result_even) = v_e_merged;
    *reinterpret_cast<uint32_t*>(result_odd)  = v_o_merged;
}

__device__ __forceinline__
void optimized_4perm(uint32_t q, int8_t* result_even, int8_t* result_odd) {
    constexpr uint32_t pos_hi = 0x0C080604u;
    constexpr uint32_t pos_lo = 0x03020100u;
    constexpr uint32_t neg_hi = 0xF4F8FAFCu;
    constexpr uint32_t neg_lo = 0xFDFEFF00u;

    uint32_t q_even = q & 0x0F0F0F0Fu;
    uint32_t q_odd  = (q >> 4) & 0x0F0F0F0Fu;

    uint32_t idx7_e = q_even & 0x07070707u;
    uint32_t idx7_o = q_odd  & 0x07070707u;

    uint32_t oob_e = (q_even << 4) & 0x80808080u;
    uint32_t oob_o = (q_odd  << 4) & 0x80808080u;

    uint32_t pos_idx_e = idx7_e | oob_e;
    uint32_t pos_idx_o = idx7_o | oob_o;

    uint32_t neg_idx_e = idx7_e | (oob_e ^ 0x80808080u);
    uint32_t neg_idx_o = idx7_o | (oob_o ^ 0x80808080u);

    uint32_t pos_e, neg_e, pos_o, neg_o;
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(pos_e) : "v"(pos_hi), "v"(pos_lo), "v"(pos_idx_e));
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(neg_e) : "v"(neg_hi), "v"(neg_lo), "v"(neg_idx_e));
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(pos_o) : "v"(pos_hi), "v"(pos_lo), "v"(pos_idx_o));
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(neg_o) : "v"(neg_hi), "v"(neg_lo), "v"(neg_idx_o));

    *reinterpret_cast<uint32_t*>(result_even) = pos_e | neg_e;
    *reinterpret_cast<uint32_t*>(result_odd)  = pos_o | neg_o;
}

__global__ void debug_kernel(uint32_t* out) {
    if (threadIdx.x != 0) return;

    uint32_t tests[] = {0x00000000, 0x01010101, 0x08080808, 0x09090909,
                        0x0F0F0F0F, 0x07070707, 0xFFFFFFFF, 0x12345678};

    for (int i = 0; i < 8; i++) {
        uint32_t q = tests[i];
        int8_t base_e[4], base_o[4];
        int8_t opt_e[4], opt_o[4];

        baseline_6perm(q, base_e, base_o);
        optimized_4perm(q, opt_e, opt_o);

        out[i*6 + 0] = q;
        out[i*6 + 1] = *reinterpret_cast<uint32_t*>(base_e);
        out[i*6 + 2] = *reinterpret_cast<uint32_t*>(base_o);
        out[i*6 + 3] = *reinterpret_cast<uint32_t*>(opt_e);
        out[i*6 + 4] = *reinterpret_cast<uint32_t*>(opt_o);
        out[i*6 + 5] = (out[i*6+1] == out[i*6+3] && out[i*6+2] == out[i*6+4]) ? 1 : 0;
    }
}

int main() {
    uint32_t* d_out;
    hipMalloc(&d_out, 48 * sizeof(uint32_t));
    debug_kernel<<<1, 1>>>(d_out);
    hipDeviceSynchronize();

    uint32_t h_out[48];
    hipMemcpy(h_out, d_out, 48 * sizeof(uint32_t), hipMemcpyDeviceToHost);

    printf("Debug Comparison: Baseline vs Optimized\n");
    printf("========================================\n\n");

    for (int i = 0; i < 8; i++) {
        printf("q = 0x%08X:\n", h_out[i*6 + 0]);
        printf("  baseline_even = 0x%08X, baseline_odd = 0x%08X\n",
               h_out[i*6 + 1], h_out[i*6 + 2]);
        printf("  optimize_even = 0x%08X, optimize_odd = 0x%08X\n",
               h_out[i*6 + 3], h_out[i*6 + 4]);
        printf("  Match: %s\n\n", h_out[i*6 + 5] ? "YES" : "NO");
    }

    hipFree(d_out);
    return 0;
}
