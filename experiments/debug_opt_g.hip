// Debug Opt G (Magnitude + Sign compute)
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>

__device__ __forceinline__
void ref_lookup(uint32_t q, int8_t* result_even, int8_t* result_odd) {
    constexpr uint32_t pos_tbl = 0x0C080604u;
    constexpr uint32_t pos_tbl2 = 0x03020100u;
    constexpr uint32_t neg_tbl = 0xF4F8FAFCu;
    constexpr uint32_t neg_tbl2 = 0xFDFEFF00u;

    uint32_t q_even = q & 0x0F0F0F0Fu;
    uint32_t q_odd  = (q >> 4) & 0x0F0F0F0Fu;

    uint32_t pos_idx_even = (q_even & 0x07070707u) | ((q_even & 0x08080808u) << 4);
    uint32_t pos_idx_odd  = (q_odd  & 0x07070707u) | ((q_odd  & 0x08080808u) << 4);

    uint32_t neg_idx_temp_even = q_even ^ 0x08080808u;
    uint32_t neg_idx_temp_odd  = q_odd  ^ 0x08080808u;
    uint32_t neg_idx_even = (neg_idx_temp_even & 0x07070707u) | ((neg_idx_temp_even & 0x08080808u) << 4);
    uint32_t neg_idx_odd  = (neg_idx_temp_odd  & 0x07070707u) | ((neg_idx_temp_odd  & 0x08080808u) << 4);

    uint32_t pos_even, pos_odd, neg_even, neg_odd;
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(pos_even) : "v"(pos_tbl), "v"(pos_tbl2), "v"(pos_idx_even));
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(neg_even) : "v"(neg_tbl), "v"(neg_tbl2), "v"(neg_idx_even));
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(pos_odd)  : "v"(pos_tbl), "v"(pos_tbl2), "v"(pos_idx_odd));
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(neg_odd)  : "v"(neg_tbl), "v"(neg_tbl2), "v"(neg_idx_odd));

    *reinterpret_cast<uint32_t*>(result_even) = pos_even | neg_even;
    *reinterpret_cast<uint32_t*>(result_odd)  = pos_odd  | neg_odd;
}

__device__ __forceinline__
void opt_g_debug(uint32_t q, int8_t* result_even, int8_t* result_odd,
                 uint32_t* dbg_mag_e, uint32_t* dbg_neg_e, uint32_t* dbg_sign_e, uint32_t* dbg_mask_e) {
    constexpr uint32_t mag_tbl = 0x0C080604u;
    constexpr uint32_t mag_tbl2 = 0x03020100u;

    uint32_t q_even = q & 0x0F0F0F0Fu;
    uint32_t q_odd  = (q >> 4) & 0x0F0F0F0Fu;
    uint32_t idx_even = q_even & 0x07070707u;
    uint32_t idx_odd  = q_odd & 0x07070707u;

    uint32_t mag_even, mag_odd;
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(mag_even) : "v"(mag_tbl), "v"(mag_tbl2), "v"(idx_even));
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(mag_odd)  : "v"(mag_tbl), "v"(mag_tbl2), "v"(idx_odd));

    uint32_t neg_mag_even = (~mag_even) + 0x01010101u;
    uint32_t neg_mag_odd  = (~mag_odd)  + 0x01010101u;

    uint32_t sign_even = (q_even >> 3) & 0x01010101u;
    uint32_t sign_odd  = (q_odd  >> 3) & 0x01010101u;

    constexpr uint32_t expand_tbl = 0x0000FF00u;
    uint32_t mask_even, mask_odd;
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(mask_even) : "v"(expand_tbl), "v"(expand_tbl), "v"(sign_even));
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(mask_odd)  : "v"(expand_tbl), "v"(expand_tbl), "v"(sign_odd));

    uint32_t res_even, res_odd;
    asm volatile("v_bfi_b32 %0, %1, %2, %3" : "=v"(res_even) : "v"(mask_even), "v"(neg_mag_even), "v"(mag_even));
    asm volatile("v_bfi_b32 %0, %1, %2, %3" : "=v"(res_odd)  : "v"(mask_odd),  "v"(neg_mag_odd),  "v"(mag_odd));

    *reinterpret_cast<uint32_t*>(result_even) = res_even;
    *reinterpret_cast<uint32_t*>(result_odd)  = res_odd;

    *dbg_mag_e = mag_even;
    *dbg_neg_e = neg_mag_even;
    *dbg_sign_e = sign_even;
    *dbg_mask_e = mask_even;
}

__global__ void debug_kernel(uint32_t* out) {
    // Test specific values
    uint32_t test_cases[] = {
        0x00000000,  // All 0s
        0x08080808,  // All sign=1, idx=0 -> should give 0
        0x01010101,  // idx=1, sign=0 -> should give 1
        0x09090909,  // idx=1, sign=1 -> should give -1 = 0xFF
        0x0F0F0F0F,  // idx=7, sign=1 -> should give -12 = 0xF4
        0x07070707,  // idx=7, sign=0 -> should give 12 = 0x0C
    };

    for (int i = 0; i < 6; i++) {
        uint32_t q = test_cases[i];
        int8_t ref_e[4], ref_o[4], opt_e[4], opt_o[4];
        uint32_t dbg_mag, dbg_neg, dbg_sign, dbg_mask;

        ref_lookup(q, ref_e, ref_o);
        opt_g_debug(q, opt_e, opt_o, &dbg_mag, &dbg_neg, &dbg_sign, &dbg_mask);

        uint32_t ref_ve = *reinterpret_cast<uint32_t*>(ref_e);
        uint32_t opt_ve = *reinterpret_cast<uint32_t*>(opt_e);

        out[i*10 + 0] = q;
        out[i*10 + 1] = ref_ve;
        out[i*10 + 2] = opt_ve;
        out[i*10 + 3] = dbg_mag;
        out[i*10 + 4] = dbg_neg;
        out[i*10 + 5] = dbg_sign;
        out[i*10 + 6] = dbg_mask;
        out[i*10 + 7] = ref_ve == opt_ve ? 1 : 0;  // Match?
        out[i*10 + 8] = q & 0x0F0F0F0F;  // q_even
        out[i*10 + 9] = (q & 0x0F0F0F0F) & 0x07070707;  // idx_even
    }
}

int main() {
    uint32_t* d_out;
    hipMalloc(&d_out, 60 * sizeof(uint32_t));

    debug_kernel<<<1, 1>>>(d_out);
    hipDeviceSynchronize();

    uint32_t h_out[60];
    hipMemcpy(h_out, d_out, 60 * sizeof(uint32_t), hipMemcpyDeviceToHost);

    printf("Debug Opt G Correctness\n");
    printf("========================\n\n");

    const char* labels[] = {"All 0", "Sign=1,idx=0", "idx=1,sign=0", "idx=1,sign=1", "idx=7,sign=1", "idx=7,sign=0"};
    const char* expected[] = {"0x00", "0x00", "0x01", "0xFF", "0xF4", "0x0C"};

    for (int i = 0; i < 6; i++) {
        printf("Test %d: %s (expect %s per byte)\n", i, labels[i], expected[i]);
        printf("  q       = 0x%08X\n", h_out[i*10 + 0]);
        printf("  q_even  = 0x%08X\n", h_out[i*10 + 8]);
        printf("  idx_even= 0x%08X\n", h_out[i*10 + 9]);
        printf("  mag     = 0x%08X\n", h_out[i*10 + 3]);
        printf("  neg_mag = 0x%08X\n", h_out[i*10 + 4]);
        printf("  sign    = 0x%08X\n", h_out[i*10 + 5]);
        printf("  mask    = 0x%08X\n", h_out[i*10 + 6]);
        printf("  REF     = 0x%08X\n", h_out[i*10 + 1]);
        printf("  OPT     = 0x%08X\n", h_out[i*10 + 2]);
        printf("  Match   = %s\n\n", h_out[i*10 + 7] ? "YES" : "NO");
    }

    hipFree(d_out);
    return 0;
}
