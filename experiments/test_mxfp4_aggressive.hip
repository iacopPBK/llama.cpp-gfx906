// MXFP4 Aggressive Optimization Tests for GFX906
// Building on OOB zeroing success, pushing further
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
#include <chrono>

//=============================================================================
// Reference: Best so far (OOB Zeroing from previous tests)
//=============================================================================
__device__ __forceinline__
void ref_oob_zeroing(uint32_t q, int8_t* result_even, int8_t* result_odd) {
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

//=============================================================================
// Optimization A: Single-pass index computation with v_bfe_u32
// v_bfe_u32 extracts bit field - might help with nibble processing
//=============================================================================
__device__ __forceinline__
void opt_a_bfe(uint32_t q, int8_t* result_even, int8_t* result_odd) {
    constexpr uint32_t pos_tbl = 0x0C080604u;
    constexpr uint32_t pos_tbl2 = 0x03020100u;
    constexpr uint32_t neg_tbl = 0xF4F8FAFCu;
    constexpr uint32_t neg_tbl2 = 0xFDFEFF00u;

    // Extract nibbles: even at positions 0,8,16,24; odd at 4,12,20,28
    uint32_t q_even = q & 0x0F0F0F0Fu;
    uint32_t q_odd;
    asm volatile("v_lshrrev_b32 %0, 4, %1" : "=v"(q_odd) : "v"(q));
    q_odd &= 0x0F0F0F0Fu;

    // Combined index computation using v_and_or_b32
    // pos_idx = (q & 0x07) | ((q & 0x08) << 4)
    // Rewrite: pos_idx = (q & 0x07) | (q & 0x80) (if we shift 0x08->0x80 first)
    // Actually: (q << 4) & 0x80808080 gives us bit3 at bit7 position

    uint32_t sign_shifted_even, sign_shifted_odd;
    asm volatile("v_lshlrev_b32 %0, 4, %1" : "=v"(sign_shifted_even) : "v"(q_even));
    asm volatile("v_lshlrev_b32 %0, 4, %1" : "=v"(sign_shifted_odd)  : "v"(q_odd));

    uint32_t pos_idx_even, pos_idx_odd;
    asm volatile("v_and_or_b32 %0, %1, %2, %3" : "=v"(pos_idx_even)
                 : "v"(sign_shifted_even), "v"(0x80808080u), "v"(q_even & 0x07070707u));
    asm volatile("v_and_or_b32 %0, %1, %2, %3" : "=v"(pos_idx_odd)
                 : "v"(sign_shifted_odd), "v"(0x80808080u), "v"(q_odd & 0x07070707u));

    // For negative: flip sign bit first, then same logic
    uint32_t q_even_neg = q_even ^ 0x08080808u;
    uint32_t q_odd_neg  = q_odd  ^ 0x08080808u;

    uint32_t sign_shifted_even_neg, sign_shifted_odd_neg;
    asm volatile("v_lshlrev_b32 %0, 4, %1" : "=v"(sign_shifted_even_neg) : "v"(q_even_neg));
    asm volatile("v_lshlrev_b32 %0, 4, %1" : "=v"(sign_shifted_odd_neg)  : "v"(q_odd_neg));

    uint32_t neg_idx_even, neg_idx_odd;
    asm volatile("v_and_or_b32 %0, %1, %2, %3" : "=v"(neg_idx_even)
                 : "v"(sign_shifted_even_neg), "v"(0x80808080u), "v"(q_even_neg & 0x07070707u));
    asm volatile("v_and_or_b32 %0, %1, %2, %3" : "=v"(neg_idx_odd)
                 : "v"(sign_shifted_odd_neg), "v"(0x80808080u), "v"(q_odd_neg & 0x07070707u));

    uint32_t pos_even, pos_odd, neg_even, neg_odd;
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(pos_even) : "v"(pos_tbl), "v"(pos_tbl2), "v"(pos_idx_even));
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(neg_even) : "v"(neg_tbl), "v"(neg_tbl2), "v"(neg_idx_even));
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(pos_odd)  : "v"(pos_tbl), "v"(pos_tbl2), "v"(pos_idx_odd));
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(neg_odd)  : "v"(neg_tbl), "v"(neg_tbl2), "v"(neg_idx_odd));

    *reinterpret_cast<uint32_t*>(result_even) = pos_even | neg_even;
    *reinterpret_cast<uint32_t*>(result_odd)  = pos_odd  | neg_odd;
}

//=============================================================================
// Optimization B: Shared sign extraction before even/odd split
// Extract all sign bits from original q before splitting
//=============================================================================
__device__ __forceinline__
void opt_b_shared_sign(uint32_t q, int8_t* result_even, int8_t* result_odd) {
    constexpr uint32_t pos_tbl = 0x0C080604u;
    constexpr uint32_t pos_tbl2 = 0x03020100u;
    constexpr uint32_t neg_tbl = 0xF4F8FAFCu;
    constexpr uint32_t neg_tbl2 = 0xFDFEFF00u;

    // Pre-compute sign-related masks from full q before splitting
    // Even nibble signs: bits 3, 11, 19, 27 -> 0x08080808
    // Odd nibble signs:  bits 7, 15, 23, 31 -> 0x80808080
    uint32_t sign_even_bits = q & 0x08080808u;  // Sign bits for even nibbles
    uint32_t sign_odd_bits  = q & 0x80808080u;  // Sign bits for odd nibbles (pre-shifted!)

    // Now extract nibbles
    uint32_t q_even = q & 0x0F0F0F0Fu;
    uint32_t q_odd  = (q >> 4) & 0x0F0F0F0Fu;

    // For even: pos_idx = (q_even & 0x07) | (sign_even << 4)
    uint32_t pos_idx_even = (q_even & 0x07070707u) | (sign_even_bits << 4);
    // For odd: pos_idx = (q_odd & 0x07) | sign_odd (already at bit 7!)
    uint32_t pos_idx_odd = (q_odd & 0x07070707u) | sign_odd_bits;

    // Negative indices: flip sign, then same pattern
    uint32_t neg_idx_even = ((q_even ^ 0x08080808u) & 0x07070707u) | ((sign_even_bits ^ 0x08080808u) << 4);
    uint32_t neg_idx_odd  = ((q_odd  ^ 0x08080808u) & 0x07070707u) | (sign_odd_bits ^ 0x80808080u);

    uint32_t pos_even, pos_odd, neg_even, neg_odd;
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(pos_even) : "v"(pos_tbl), "v"(pos_tbl2), "v"(pos_idx_even));
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(neg_even) : "v"(neg_tbl), "v"(neg_tbl2), "v"(neg_idx_even));
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(pos_odd)  : "v"(pos_tbl), "v"(pos_tbl2), "v"(pos_idx_odd));
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(neg_odd)  : "v"(neg_tbl), "v"(neg_tbl2), "v"(neg_idx_odd));

    *reinterpret_cast<uint32_t*>(result_even) = pos_even | neg_even;
    *reinterpret_cast<uint32_t*>(result_odd)  = pos_odd  | neg_odd;
}

//=============================================================================
// Optimization C: Merged nibble+index in single v_bfi_b32
// v_bfi_b32 D, S0, S1, S2 : D = (S0 & S1) | (~S0 & S2)
//=============================================================================
__device__ __forceinline__
void opt_c_bfi_merge(uint32_t q, int8_t* result_even, int8_t* result_odd) {
    constexpr uint32_t pos_tbl = 0x0C080604u;
    constexpr uint32_t pos_tbl2 = 0x03020100u;
    constexpr uint32_t neg_tbl = 0xF4F8FAFCu;
    constexpr uint32_t neg_tbl2 = 0xFDFEFF00u;

    uint32_t q_even = q & 0x0F0F0F0Fu;
    uint32_t q_odd  = (q >> 4) & 0x0F0F0F0Fu;

    // Compute both pos and neg index at once using BFI
    // We want: pos_idx = (idx & 0x07) | ((idx & 0x08) ? 0x80 : 0)
    // BFI pattern: (mask & val1) | (~mask & val2)
    // If we have idx & 0x08 expanded to full byte mask, we can select

    // First, get magnitude index
    uint32_t mag_even = q_even & 0x07070707u;
    uint32_t mag_odd  = q_odd & 0x07070707u;

    // Get sign bits shifted to bit 7 position for OOB marking
    uint32_t oob_even = (q_even & 0x08080808u) << 4;  // bit3 -> bit7
    uint32_t oob_odd  = (q_odd & 0x08080808u) << 4;

    // pos_idx: magnitude OR'd with OOB mask (high indices get 0x80)
    uint32_t pos_idx_even = mag_even | oob_even;
    uint32_t pos_idx_odd  = mag_odd  | oob_odd;

    // neg_idx: flip sign interpretation
    uint32_t neg_idx_even = mag_even | (oob_even ^ 0x80808080u);
    uint32_t neg_idx_odd  = mag_odd  | (oob_odd ^ 0x80808080u);

    uint32_t pos_even, pos_odd, neg_even, neg_odd;
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(pos_even) : "v"(pos_tbl), "v"(pos_tbl2), "v"(pos_idx_even));
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(neg_even) : "v"(neg_tbl), "v"(neg_tbl2), "v"(neg_idx_even));
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(pos_odd)  : "v"(pos_tbl), "v"(pos_tbl2), "v"(pos_idx_odd));
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(neg_odd)  : "v"(neg_tbl), "v"(neg_tbl2), "v"(neg_idx_odd));

    *reinterpret_cast<uint32_t*>(result_even) = pos_even | neg_even;
    *reinterpret_cast<uint32_t*>(result_odd)  = pos_odd  | neg_odd;
}

//=============================================================================
// Optimization D: Direct byte manipulation with v_alignbit_b32
// Process both even and odd simultaneously with clever bit arrangement
//=============================================================================
__device__ __forceinline__
void opt_d_alignbit(uint32_t q, int8_t* result_even, int8_t* result_odd) {
    constexpr uint32_t pos_tbl = 0x0C080604u;
    constexpr uint32_t pos_tbl2 = 0x03020100u;
    constexpr uint32_t neg_tbl = 0xF4F8FAFCu;
    constexpr uint32_t neg_tbl2 = 0xFDFEFF00u;

    // Use v_alignbit to create different views of q
    // v_alignbit_b32 D, S0, S1, S2: D = (S0 << (32-S2)) | (S1 >> S2)
    // This can shift bits across register boundary

    uint32_t q_even = q & 0x0F0F0F0Fu;
    uint32_t q_odd;
    // Use alignbit instead of lshr: alignbit(0, q, 4) = (0 << 28) | (q >> 4) = q >> 4
    asm volatile("v_alignbit_b32 %0, %1, %2, 4" : "=v"(q_odd) : "v"(0u), "v"(q));
    q_odd &= 0x0F0F0F0Fu;

    // Standard OOB processing from here
    uint32_t mag_even = q_even & 0x07070707u;
    uint32_t mag_odd  = q_odd & 0x07070707u;
    uint32_t oob_even = (q_even << 4) & 0x80808080u;
    uint32_t oob_odd  = (q_odd << 4) & 0x80808080u;

    uint32_t pos_idx_even = mag_even | oob_even;
    uint32_t pos_idx_odd  = mag_odd  | oob_odd;
    uint32_t neg_idx_even = mag_even | (oob_even ^ 0x80808080u);
    uint32_t neg_idx_odd  = mag_odd  | (oob_odd ^ 0x80808080u);

    uint32_t pos_even, pos_odd, neg_even, neg_odd;
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(pos_even) : "v"(pos_tbl), "v"(pos_tbl2), "v"(pos_idx_even));
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(neg_even) : "v"(neg_tbl), "v"(neg_tbl2), "v"(neg_idx_even));
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(pos_odd)  : "v"(pos_tbl), "v"(pos_tbl2), "v"(pos_idx_odd));
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(neg_odd)  : "v"(neg_tbl), "v"(neg_tbl2), "v"(neg_idx_odd));

    *reinterpret_cast<uint32_t*>(result_even) = pos_even | neg_even;
    *reinterpret_cast<uint32_t*>(result_odd)  = pos_odd  | neg_odd;
}

//=============================================================================
// Optimization E: MINIMAL version - fewest possible operations
// Focus on reducing total instruction count
//=============================================================================
__device__ __forceinline__
void opt_e_minimal(uint32_t q, int8_t* result_even, int8_t* result_odd) {
    // Key insight: We can compute neg_idx from pos_idx with just XOR
    // Because: neg_idx = ((q ^ 0x08) & 0x07) | (((q ^ 0x08) & 0x08) << 4)
    //        = (q & 0x07) | (((q ^ 0x08) & 0x08) << 4)
    //        = (q & 0x07) | ((0x08 ^ (q & 0x08)) << 4)
    //        = pos_idx XOR 0x80808080 (if bit3 was 0) OR pos_idx XOR 0x80808080 (if bit3 was 1)
    // Actually: neg_idx = pos_idx XOR 0x80808080 (always!)

    constexpr uint32_t pos_tbl = 0x0C080604u;
    constexpr uint32_t pos_tbl2 = 0x03020100u;
    constexpr uint32_t neg_tbl = 0xF4F8FAFCu;
    constexpr uint32_t neg_tbl2 = 0xFDFEFF00u;

    uint32_t q_even = q & 0x0F0F0F0Fu;
    uint32_t q_odd  = (q >> 4) & 0x0F0F0F0Fu;

    // pos_idx = (q & 0x07) | ((q & 0x08) << 4)
    // Use v_and_or: (q_shifted & 0x80) | (q & 0x07)
    uint32_t q_shifted_even = q_even << 4;
    uint32_t q_shifted_odd  = q_odd << 4;

    uint32_t pos_idx_even, pos_idx_odd;
    asm volatile("v_and_or_b32 %0, %1, %2, %3" : "=v"(pos_idx_even)
                 : "v"(q_shifted_even), "v"(0x80808080u), "v"(q_even & 0x07070707u));
    asm volatile("v_and_or_b32 %0, %1, %2, %3" : "=v"(pos_idx_odd)
                 : "v"(q_shifted_odd), "v"(0x80808080u), "v"(q_odd & 0x07070707u));

    // neg_idx = pos_idx XOR 0x80808080
    uint32_t neg_idx_even = pos_idx_even ^ 0x80808080u;
    uint32_t neg_idx_odd  = pos_idx_odd ^ 0x80808080u;

    uint32_t pos_even, pos_odd, neg_even, neg_odd;
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(pos_even) : "v"(pos_tbl), "v"(pos_tbl2), "v"(pos_idx_even));
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(neg_even) : "v"(neg_tbl), "v"(neg_tbl2), "v"(neg_idx_even));
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(pos_odd)  : "v"(pos_tbl), "v"(pos_tbl2), "v"(pos_idx_odd));
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(neg_odd)  : "v"(neg_tbl), "v"(neg_tbl2), "v"(neg_idx_odd));

    *reinterpret_cast<uint32_t*>(result_even) = pos_even | neg_even;
    *reinterpret_cast<uint32_t*>(result_odd)  = pos_odd  | neg_odd;
}

//=============================================================================
// Optimization F: Process 64-bits at once (2 words -> 16 nibbles)
// Reduces loop overhead by processing more data per iteration
//=============================================================================
__device__ __forceinline__
void opt_f_64bit(uint32_t q0, uint32_t q1, int8_t* result0_even, int8_t* result0_odd,
                 int8_t* result1_even, int8_t* result1_odd) {
    constexpr uint32_t pos_tbl = 0x0C080604u;
    constexpr uint32_t pos_tbl2 = 0x03020100u;
    constexpr uint32_t neg_tbl = 0xF4F8FAFCu;
    constexpr uint32_t neg_tbl2 = 0xFDFEFF00u;

    // Process both words using shared table constants
    // Word 0
    uint32_t q0_even = q0 & 0x0F0F0F0Fu;
    uint32_t q0_odd  = (q0 >> 4) & 0x0F0F0F0Fu;
    uint32_t q0_shifted_even = q0_even << 4;
    uint32_t q0_shifted_odd  = q0_odd << 4;

    uint32_t pos_idx0_even, pos_idx0_odd;
    asm volatile("v_and_or_b32 %0, %1, %2, %3" : "=v"(pos_idx0_even)
                 : "v"(q0_shifted_even), "v"(0x80808080u), "v"(q0_even & 0x07070707u));
    asm volatile("v_and_or_b32 %0, %1, %2, %3" : "=v"(pos_idx0_odd)
                 : "v"(q0_shifted_odd), "v"(0x80808080u), "v"(q0_odd & 0x07070707u));

    uint32_t neg_idx0_even = pos_idx0_even ^ 0x80808080u;
    uint32_t neg_idx0_odd  = pos_idx0_odd ^ 0x80808080u;

    // Word 1
    uint32_t q1_even = q1 & 0x0F0F0F0Fu;
    uint32_t q1_odd  = (q1 >> 4) & 0x0F0F0F0Fu;
    uint32_t q1_shifted_even = q1_even << 4;
    uint32_t q1_shifted_odd  = q1_odd << 4;

    uint32_t pos_idx1_even, pos_idx1_odd;
    asm volatile("v_and_or_b32 %0, %1, %2, %3" : "=v"(pos_idx1_even)
                 : "v"(q1_shifted_even), "v"(0x80808080u), "v"(q1_even & 0x07070707u));
    asm volatile("v_and_or_b32 %0, %1, %2, %3" : "=v"(pos_idx1_odd)
                 : "v"(q1_shifted_odd), "v"(0x80808080u), "v"(q1_odd & 0x07070707u));

    uint32_t neg_idx1_even = pos_idx1_even ^ 0x80808080u;
    uint32_t neg_idx1_odd  = pos_idx1_odd ^ 0x80808080u;

    // All 8 perms
    uint32_t pos0e, neg0e, pos0o, neg0o, pos1e, neg1e, pos1o, neg1o;
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(pos0e) : "v"(pos_tbl), "v"(pos_tbl2), "v"(pos_idx0_even));
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(neg0e) : "v"(neg_tbl), "v"(neg_tbl2), "v"(neg_idx0_even));
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(pos0o) : "v"(pos_tbl), "v"(pos_tbl2), "v"(pos_idx0_odd));
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(neg0o) : "v"(neg_tbl), "v"(neg_tbl2), "v"(neg_idx0_odd));
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(pos1e) : "v"(pos_tbl), "v"(pos_tbl2), "v"(pos_idx1_even));
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(neg1e) : "v"(neg_tbl), "v"(neg_tbl2), "v"(neg_idx1_even));
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(pos1o) : "v"(pos_tbl), "v"(pos_tbl2), "v"(pos_idx1_odd));
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(neg1o) : "v"(neg_tbl), "v"(neg_tbl2), "v"(neg_idx1_odd));

    *reinterpret_cast<uint32_t*>(result0_even) = pos0e | neg0e;
    *reinterpret_cast<uint32_t*>(result0_odd)  = pos0o | neg0o;
    *reinterpret_cast<uint32_t*>(result1_even) = pos1e | neg1e;
    *reinterpret_cast<uint32_t*>(result1_odd)  = pos1o | neg1o;
}

//=============================================================================
// Optimization G: Single lookup path with computed sign application
// Lookup magnitude only, then apply sign via XOR + correction
//=============================================================================
__device__ __forceinline__
void opt_g_mag_then_sign(uint32_t q, int8_t* result_even, int8_t* result_odd) {
    // Lookup positive magnitudes only, then negate if needed
    // For bytes 0-3 (values 0,1,2,3): neg = -x = (~x) + 1
    // For bytes 4-12: need lookup

    // Key insight: XOR with 0xFF then add 1 gives negation, BUT carries across bytes
    // Unless all values are < 128, then no carry!
    // Max magnitude = 12 = 0x0C, so ~12 = 0xF3, +1 = 0xF4 (no carry)
    // This WORKS for our magnitudes!

    constexpr uint32_t mag_tbl = 0x0C080604u;  // {4,6,8,12} for indices 4-7
    constexpr uint32_t mag_tbl2 = 0x03020100u; // {0,1,2,3} for indices 0-3

    uint32_t q_even = q & 0x0F0F0F0Fu;
    uint32_t q_odd  = (q >> 4) & 0x0F0F0F0Fu;
    uint32_t idx_even = q_even & 0x07070707u;
    uint32_t idx_odd  = q_odd & 0x07070707u;

    // Single perm for magnitude
    uint32_t mag_even, mag_odd;
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(mag_even) : "v"(mag_tbl), "v"(mag_tbl2), "v"(idx_even));
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(mag_odd)  : "v"(mag_tbl), "v"(mag_tbl2), "v"(idx_odd));

    // Compute negative magnitude: ~mag + 1 (safe because max mag = 12 < 128)
    uint32_t neg_mag_even = (~mag_even) + 0x01010101u;
    uint32_t neg_mag_odd  = (~mag_odd)  + 0x01010101u;

    // Select based on sign bit (bit 3)
    // Expand sign bit to full byte: use perm with {0x00, 0xFF} table
    // Or use BFI to select

    uint32_t sign_even = (q_even >> 3) & 0x01010101u;  // 0 or 1 per byte
    uint32_t sign_odd  = (q_odd  >> 3) & 0x01010101u;

    // Expand to mask: sign * 0xFF = 0x00 or 0xFF
    // Use: 0 - sign = 0x00 or 0xFFFFFFFF per byte... no, carries!
    // Use perm: {0x00, 0xFF, x, x, x, x, x, x}
    constexpr uint32_t expand_tbl = 0x0000FF00u;
    uint32_t mask_even, mask_odd;
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(mask_even) : "v"(expand_tbl), "v"(expand_tbl), "v"(sign_even));
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(mask_odd)  : "v"(expand_tbl), "v"(expand_tbl), "v"(sign_odd));

    // Select: (neg & mask) | (pos & ~mask) = BFI
    uint32_t res_even, res_odd;
    asm volatile("v_bfi_b32 %0, %1, %2, %3" : "=v"(res_even) : "v"(mask_even), "v"(neg_mag_even), "v"(mag_even));
    asm volatile("v_bfi_b32 %0, %1, %2, %3" : "=v"(res_odd)  : "v"(mask_odd),  "v"(neg_mag_odd),  "v"(mag_odd));

    *reinterpret_cast<uint32_t*>(result_even) = res_even;
    *reinterpret_cast<uint32_t*>(result_odd)  = res_odd;
}

//=============================================================================
// Benchmark kernels
//=============================================================================

__global__ void benchmark_ref(uint32_t* __restrict__ out, const uint32_t* __restrict__ in, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    uint32_t q = in[idx];
    int8_t result_even[4], result_odd[4];

    #pragma unroll
    for (int i = 0; i < 100; i++) {
        ref_oob_zeroing(q, result_even, result_odd);
        q ^= *reinterpret_cast<uint32_t*>(result_even) ^ *reinterpret_cast<uint32_t*>(result_odd);
    }

    out[idx*2]   = *reinterpret_cast<uint32_t*>(result_even);
    out[idx*2+1] = *reinterpret_cast<uint32_t*>(result_odd);
}

__global__ void benchmark_opt_c(uint32_t* __restrict__ out, const uint32_t* __restrict__ in, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    uint32_t q = in[idx];
    int8_t result_even[4], result_odd[4];

    #pragma unroll
    for (int i = 0; i < 100; i++) {
        opt_c_bfi_merge(q, result_even, result_odd);
        q ^= *reinterpret_cast<uint32_t*>(result_even) ^ *reinterpret_cast<uint32_t*>(result_odd);
    }

    out[idx*2]   = *reinterpret_cast<uint32_t*>(result_even);
    out[idx*2+1] = *reinterpret_cast<uint32_t*>(result_odd);
}

__global__ void benchmark_opt_e(uint32_t* __restrict__ out, const uint32_t* __restrict__ in, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    uint32_t q = in[idx];
    int8_t result_even[4], result_odd[4];

    #pragma unroll
    for (int i = 0; i < 100; i++) {
        opt_e_minimal(q, result_even, result_odd);
        q ^= *reinterpret_cast<uint32_t*>(result_even) ^ *reinterpret_cast<uint32_t*>(result_odd);
    }

    out[idx*2]   = *reinterpret_cast<uint32_t*>(result_even);
    out[idx*2+1] = *reinterpret_cast<uint32_t*>(result_odd);
}

__global__ void benchmark_opt_g(uint32_t* __restrict__ out, const uint32_t* __restrict__ in, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    uint32_t q = in[idx];
    int8_t result_even[4], result_odd[4];

    #pragma unroll
    for (int i = 0; i < 100; i++) {
        opt_g_mag_then_sign(q, result_even, result_odd);
        q ^= *reinterpret_cast<uint32_t*>(result_even) ^ *reinterpret_cast<uint32_t*>(result_odd);
    }

    out[idx*2]   = *reinterpret_cast<uint32_t*>(result_even);
    out[idx*2+1] = *reinterpret_cast<uint32_t*>(result_odd);
}

//=============================================================================
// Verification
//=============================================================================

__global__ void verify_all(int* errors) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 65536) return;

    uint32_t q = idx | (idx << 16);
    int8_t ref_e[4], ref_o[4];
    int8_t c_e[4], c_o[4];
    int8_t e_e[4], e_o[4];
    int8_t g_e[4], g_o[4];

    ref_oob_zeroing(q, ref_e, ref_o);
    opt_c_bfi_merge(q, c_e, c_o);
    opt_e_minimal(q, e_e, e_o);
    opt_g_mag_then_sign(q, g_e, g_o);

    uint32_t ref_ve = *reinterpret_cast<uint32_t*>(ref_e);
    uint32_t ref_vo = *reinterpret_cast<uint32_t*>(ref_o);

    if (*reinterpret_cast<uint32_t*>(c_e) != ref_ve || *reinterpret_cast<uint32_t*>(c_o) != ref_vo) {
        atomicAdd(&errors[0], 1);
    }
    if (*reinterpret_cast<uint32_t*>(e_e) != ref_ve || *reinterpret_cast<uint32_t*>(e_o) != ref_vo) {
        atomicAdd(&errors[1], 1);
    }
    if (*reinterpret_cast<uint32_t*>(g_e) != ref_ve || *reinterpret_cast<uint32_t*>(g_o) != ref_vo) {
        atomicAdd(&errors[2], 1);
    }
}

//=============================================================================
// Main
//=============================================================================

int main() {
    printf("MXFP4 Aggressive Optimization Tests for GFX906\n");
    printf("===============================================\n\n");

    // Verify correctness
    printf("Phase 1: Correctness Verification\n");
    printf("---------------------------------\n");

    int* d_errors;
    hipMalloc(&d_errors, 4 * sizeof(int));
    hipMemset(d_errors, 0, 4 * sizeof(int));

    verify_all<<<256, 256>>>(d_errors);
    hipDeviceSynchronize();

    int h_errors[4];
    hipMemcpy(h_errors, d_errors, 4 * sizeof(int), hipMemcpyDeviceToHost);

    printf("Opt C (BFI merge) errors: %d\n", h_errors[0]);
    printf("Opt E (Minimal)   errors: %d\n", h_errors[1]);
    printf("Opt G (Mag+Sign)  errors: %d\n", h_errors[2]);

    if (h_errors[0] == 0 && h_errors[1] == 0 && h_errors[2] == 0) {
        printf("\n✓ All optimizations produce correct results!\n");
    } else {
        printf("\n⚠️  Some optimizations have errors!\n");
    }

    hipFree(d_errors);

    // Performance benchmark
    printf("\nPhase 2: Performance Benchmark\n");
    printf("------------------------------\n");

    const int N = 1024 * 1024;
    const int WARMUP = 5;
    const int RUNS = 20;

    uint32_t* d_in;
    uint32_t* d_out;
    hipMalloc(&d_in, N * sizeof(uint32_t));
    hipMalloc(&d_out, N * 2 * sizeof(uint32_t));

    uint32_t* h_in = new uint32_t[N];
    for (int i = 0; i < N; i++) {
        h_in[i] = rand();
    }
    hipMemcpy(d_in, h_in, N * sizeof(uint32_t), hipMemcpyHostToDevice);
    delete[] h_in;

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    auto benchmark = [&](const char* name, auto kernel) {
        for (int i = 0; i < WARMUP; i++) {
            kernel<<<gridSize, blockSize>>>(d_out, d_in, N);
        }
        hipDeviceSynchronize();

        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < RUNS; i++) {
            kernel<<<gridSize, blockSize>>>(d_out, d_in, N);
        }
        hipDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();

        double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
        double avg_ms = elapsed_ms / RUNS;
        double ops_per_sec = (N * 100.0) / (avg_ms / 1000.0);

        printf("%-25s: %.3f ms avg, %.2f B ops/sec\n", name, avg_ms, ops_per_sec / 1e9);
    };

    benchmark("Ref (OOB zeroing)", benchmark_ref);
    benchmark("Opt C (BFI merge)", benchmark_opt_c);
    benchmark("Opt E (Minimal)", benchmark_opt_e);
    benchmark("Opt G (Mag+Sign compute)", benchmark_opt_g);

    hipFree(d_in);
    hipFree(d_out);

    printf("\nDone.\n");
    return 0;
}
