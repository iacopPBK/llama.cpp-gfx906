// MXFP4 Byte Operation Optimization Tests for GFX906
// Tests various approaches to the magnitude-sign lookup problem
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
#include <chrono>

// The MXFP4 lookup table
// Index:  0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15
// Value:  0   1   2   3   4   6   8  12   0  -1  -2  -3  -4  -6  -8 -12

//=============================================================================
// GFX906 Instruction Exploration
//=============================================================================

// Check: Does v_sad_u8 give us per-byte absolute difference?
// v_sad_u8 D, S0, S1, S2 : D = SUM(|S0[i] - S1[i]|) + S2
// This sums across bytes - NOT useful for per-byte ops

// Check: v_qsad_pk_u16_u8 - Quad SAD, outputs 16-bit sums
// Not useful for our case

// Key Insight: GFX906 has NO packed 8-bit arithmetic
// We must work with v_perm_b32 and 32-bit ops

//=============================================================================
// Baseline: Current 6-perm approach (from llama.cpp)
//=============================================================================
__device__ __forceinline__
void baseline_6perm(uint32_t q, int8_t* result_even, int8_t* result_odd) {
    // Table chunks
    constexpr uint32_t tbl_lo_pos = 0x03020100u;  // {0, 1, 2, 3}
    constexpr uint32_t tbl_hi_pos = 0x0C080604u;  // {4, 6, 8, 12}
    constexpr uint32_t tbl_lo_neg = 0xFDFEFF00u;  // {0, -1, -2, -3}
    constexpr uint32_t tbl_hi_neg = 0xF4F8FAFCu;  // {-4, -6, -8, -12}

    // Extract nibbles
    uint32_t q_even = q & 0x0F0F0F0Fu;
    uint32_t q_odd  = (q >> 4) & 0x0F0F0F0Fu;

    // Mask for low 3 bits of each byte
    uint32_t idx_even = q_even & 0x07070707u;
    uint32_t idx_odd  = q_odd & 0x07070707u;

    // Even nibbles: 6 operations (2 perm + mask + perm)
    uint32_t v_e_lo, v_e_hi, v_e_merged;
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(v_e_lo) : "v"(tbl_hi_pos), "v"(tbl_lo_pos), "v"(idx_even));
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(v_e_hi) : "v"(tbl_hi_neg), "v"(tbl_lo_neg), "v"(idx_even));
    uint32_t mask_e = 0x03020100u | ((q_even & 0x08080808u) >> 1);
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(v_e_merged) : "v"(v_e_hi), "v"(v_e_lo), "v"(mask_e));

    // Odd nibbles: same pattern
    uint32_t v_o_lo, v_o_hi, v_o_merged;
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(v_o_lo) : "v"(tbl_hi_pos), "v"(tbl_lo_pos), "v"(idx_odd));
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(v_o_hi) : "v"(tbl_hi_neg), "v"(tbl_lo_neg), "v"(idx_odd));
    uint32_t mask_o = 0x03020100u | ((q_odd & 0x08080808u) >> 1);
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(v_o_merged) : "v"(v_o_hi), "v"(v_o_lo), "v"(mask_o));

    *reinterpret_cast<uint32_t*>(result_even) = v_e_merged;
    *reinterpret_cast<uint32_t*>(result_odd)  = v_o_merged;
}

//=============================================================================
// Optimization 1: Magnitude-Sign Split (2 perm + sign handling)
//=============================================================================

// Method 1A: Use v_bfi_b32 for per-byte sign application
// v_bfi_b32 D, S0, S1, S2 : D = (S0 & S1) | (~S0 & S2)
// This is bit field insert - could help with selective negation

__device__ __forceinline__
void opt1a_magnitude_sign_bfi(uint32_t q, int8_t* result_even, int8_t* result_odd) {
    // Single magnitude table using split-source trick:
    // v_perm selects bytes 0-3 from src2, bytes 4-7 from src1
    // Pack: src2 = {0,1,2,3}, src1 = {4,6,8,12}
    constexpr uint32_t mag_src2 = 0x03020100u;  // {0, 1, 2, 3} for indices 0-3
    constexpr uint32_t mag_src1 = 0x0C080604u;  // {4, 6, 8, 12} for indices 4-7

    // Extract nibbles
    uint32_t q_even = q & 0x0F0F0F0Fu;
    uint32_t q_odd  = (q >> 4) & 0x0F0F0F0Fu;

    // Mask to get magnitude index (bits 2:0 of each nibble)
    uint32_t idx_even = q_even & 0x07070707u;
    uint32_t idx_odd  = q_odd & 0x07070707u;

    // Single perm for magnitude lookup (each nibble)
    uint32_t mag_even, mag_odd;
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(mag_even) : "v"(mag_src1), "v"(mag_src2), "v"(idx_even));
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(mag_odd)  : "v"(mag_src1), "v"(mag_src2), "v"(idx_odd));

    // Extract sign bits (bit 3 of each nibble)
    uint32_t sign_even = (q_even >> 3) & 0x01010101u;
    uint32_t sign_odd  = (q_odd >> 3) & 0x01010101u;

    // Expand sign to full bytes: 0x00 or 0xFF
    // sign * 0xFF = 0 or 0xFF (but multiplication is expensive)
    // Alternative: (sign << 8) - sign = 0 or 0xFF (still has carry issues)
    // Use: 0 - sign, then AND with 0xFF per byte
    // Actually: v_sub will have carry issues...

    // Method: Lookup negative values separately, then select
    constexpr uint32_t neg_src2 = 0xFDFEFF00u;  // {0, -1, -2, -3}
    constexpr uint32_t neg_src1 = 0xF4F8FAFCu;  // {-4, -6, -8, -12}

    uint32_t neg_even, neg_odd;
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(neg_even) : "v"(neg_src1), "v"(neg_src2), "v"(idx_even));
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(neg_odd)  : "v"(neg_src1), "v"(neg_src2), "v"(idx_odd));

    // Create selection mask from sign bits
    // Expand 0x01 -> 0xFF using: -sign (but carries!) or (sign * 255) or lookup
    // Use BFI: result = (sign_mask & neg) | (~sign_mask & pos)
    // But we need sign_mask to be 0x00 or 0xFF per byte

    // Clever trick: Use subtraction with borrow isolation
    // 0 - 0 = 0, 0 - 1 = 0xFF (but carries to next byte!)
    // Instead: Use perm to expand!
    // sign_expanded = perm({0x00, 0xFF, x, x, x, x, x, x}, sign)
    constexpr uint32_t expand_src2 = 0x0000FF00u;  // byte0=0, byte1=0xFF
    uint32_t sign_mask_even, sign_mask_odd;
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(sign_mask_even) : "v"(expand_src2), "v"(expand_src2), "v"(sign_even));
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(sign_mask_odd)  : "v"(expand_src2), "v"(expand_src2), "v"(sign_odd));

    // BFI: result = (mask & neg) | (~mask & pos)
    uint32_t res_even, res_odd;
    asm volatile("v_bfi_b32 %0, %1, %2, %3" : "=v"(res_even) : "v"(sign_mask_even), "v"(neg_even), "v"(mag_even));
    asm volatile("v_bfi_b32 %0, %1, %2, %3" : "=v"(res_odd)  : "v"(sign_mask_odd),  "v"(neg_odd),  "v"(mag_odd));

    *reinterpret_cast<uint32_t*>(result_even) = res_even;
    *reinterpret_cast<uint32_t*>(result_odd)  = res_odd;
}

//=============================================================================
// Optimization 1B: Try XOR-based sign application
//=============================================================================

__device__ __forceinline__
void opt1b_magnitude_xor_sign(uint32_t q, int8_t* result_even, int8_t* result_odd) {
    // Insight: For small magnitudes, we can compute negation differently
    // -x = ~x + 1 (two's complement)
    // But what if we XOR and add separately per byte?

    // The magnitudes are: 0, 1, 2, 3, 4, 6, 8, 12
    // Their negatives are: 0, -1, -2, -3, -4, -6, -8, -12
    // In hex: 0x00, 0xFF, 0xFE, 0xFD, 0xFC, 0xFA, 0xF8, 0xF4
    //
    // XOR with 0xFF gives: 0xFF, 0xFE, 0xFD, 0xFC, 0xFB, 0xF9, 0xF7, 0xF3
    // We need to add 1 to each, but carry propagates!

    // New idea: Pre-compute (magnitude XOR 0xFF) + 1 in the negative table
    // Then select between pos and neg tables

    // This is what we already do in baseline... let's try something else

    // Idea: Can we use the PERM instruction to add 1 per byte?
    // No, perm is just selection

    // Idea: Lookup table for negation too (same as opt1a)
    constexpr uint32_t mag_src2 = 0x03020100u;
    constexpr uint32_t mag_src1 = 0x0C080604u;
    constexpr uint32_t neg_src2 = 0xFDFEFF00u;
    constexpr uint32_t neg_src1 = 0xF4F8FAFCu;

    uint32_t q_even = q & 0x0F0F0F0Fu;
    uint32_t q_odd  = (q >> 4) & 0x0F0F0F0Fu;
    uint32_t idx_even = q_even & 0x07070707u;
    uint32_t idx_odd  = q_odd & 0x07070707u;

    // Lookup both positive and negative
    uint32_t pos_even, pos_odd, neg_even, neg_odd;
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(pos_even) : "v"(mag_src1), "v"(mag_src2), "v"(idx_even));
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(neg_even) : "v"(neg_src1), "v"(neg_src2), "v"(idx_even));
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(pos_odd)  : "v"(mag_src1), "v"(mag_src2), "v"(idx_odd));
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(neg_odd)  : "v"(neg_src1), "v"(neg_src2), "v"(idx_odd));

    // Extract sign bits
    uint32_t sign_even = q_even & 0x08080808u;  // Keep bit 3 in position
    uint32_t sign_odd  = q_odd & 0x08080808u;

    // Use XOR to blend: if sign bit set, XOR swaps pos<->neg effectively
    // Actually we need conditional select, not XOR

    // Method: Create select mask via perm, then use AND/OR
    // select_mask = sign ? 0xFFFFFFFF : 0x00000000 per byte
    // But we need per-BYTE mask...

    // Alternative: Use the merge perm approach (like baseline but with fewer lookups)
    // The third perm selects between pos and neg based on sign
    uint32_t merge_mask_even = 0x03020100u | (sign_even >> 1);  // shifts bit3 to bit2 position
    uint32_t merge_mask_odd  = 0x03020100u | (sign_odd >> 1);

    uint32_t res_even, res_odd;
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(res_even) : "v"(neg_even), "v"(pos_even), "v"(merge_mask_even));
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(res_odd)  : "v"(neg_odd),  "v"(pos_odd),  "v"(merge_mask_odd));

    *reinterpret_cast<uint32_t*>(result_even) = res_even;
    *reinterpret_cast<uint32_t*>(result_odd)  = res_odd;
}

//=============================================================================
// Optimization 2: OOB Zeroing Approach
//=============================================================================

__device__ __forceinline__
void opt2_oob_zeroing(uint32_t q, int8_t* result_even, int8_t* result_odd) {
    // v_perm_b32 zeros output if index byte has bit 7 set
    // We can use this to auto-mask invalid lookups

    constexpr uint32_t pos_tbl = 0x0C080604u;  // {4, 6, 8, 12} in bytes 4-7
    constexpr uint32_t pos_tbl2 = 0x03020100u; // {0, 1, 2, 3} in bytes 0-3
    constexpr uint32_t neg_tbl = 0xF4F8FAFCu;  // {-4, -6, -8, -12}
    constexpr uint32_t neg_tbl2 = 0xFDFEFF00u; // {0, -1, -2, -3}

    uint32_t q_even = q & 0x0F0F0F0Fu;
    uint32_t q_odd  = (q >> 4) & 0x0F0F0F0Fu;

    // For positive table: indices 8-15 should be zeroed (set bit 7)
    // For negative table: indices 0-7 should be zeroed (set bit 7)

    // Positive: if bit3 set, invalidate by setting bit7
    // invalidate_mask = (q & 0x08080808) << 4 = bit3 shifted to bit7
    uint32_t pos_idx_even = (q_even & 0x07070707u) | ((q_even & 0x08080808u) << 4);
    uint32_t pos_idx_odd  = (q_odd  & 0x07070707u) | ((q_odd  & 0x08080808u) << 4);

    // Negative: if bit3 NOT set, invalidate; also need to map 8-15 to 0-7
    // XOR with 0x08 flips bit3 (maps 8-15 to 0-7 and 0-7 to 8-15)
    // Then same invalidation logic
    uint32_t neg_idx_temp_even = q_even ^ 0x08080808u;  // flip bit 3
    uint32_t neg_idx_temp_odd  = q_odd  ^ 0x08080808u;
    uint32_t neg_idx_even = (neg_idx_temp_even & 0x07070707u) | ((neg_idx_temp_even & 0x08080808u) << 4);
    uint32_t neg_idx_odd  = (neg_idx_temp_odd  & 0x07070707u) | ((neg_idx_temp_odd  & 0x08080808u) << 4);

    // Perform lookups (invalid indices return 0)
    uint32_t pos_even, pos_odd, neg_even, neg_odd;
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(pos_even) : "v"(pos_tbl), "v"(pos_tbl2), "v"(pos_idx_even));
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(neg_even) : "v"(neg_tbl), "v"(neg_tbl2), "v"(neg_idx_even));
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(pos_odd)  : "v"(pos_tbl), "v"(pos_tbl2), "v"(pos_idx_odd));
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(neg_odd)  : "v"(neg_tbl), "v"(neg_tbl2), "v"(neg_idx_odd));

    // Merge with OR (one is always 0)
    uint32_t res_even = pos_even | neg_even;
    uint32_t res_odd  = pos_odd  | neg_odd;

    *reinterpret_cast<uint32_t*>(result_even) = res_even;
    *reinterpret_cast<uint32_t*>(result_odd)  = res_odd;
}

//=============================================================================
// Optimization 3: Shared sign extraction (reduce dependency chain)
//=============================================================================

__device__ __forceinline__
void opt3_shared_sign(uint32_t q, int8_t* result_even, int8_t* result_odd) {
    // Pre-extract sign bits before splitting into even/odd
    // This can hide latency better

    constexpr uint32_t mag_src2 = 0x03020100u;
    constexpr uint32_t mag_src1 = 0x0C080604u;
    constexpr uint32_t neg_src2 = 0xFDFEFF00u;
    constexpr uint32_t neg_src1 = 0xF4F8FAFCu;

    // Extract sign bits from both nibbles simultaneously
    uint32_t sign_all = q & 0x88888888u;  // bit3 of each nibble

    // Now split into even/odd
    uint32_t q_even = q & 0x0F0F0F0Fu;
    uint32_t q_odd  = (q >> 4) & 0x0F0F0F0Fu;

    uint32_t idx_even = q_even & 0x07070707u;
    uint32_t idx_odd  = q_odd & 0x07070707u;

    // Sign bits for even nibbles: bits 3, 11, 19, 27 -> need in byte positions
    // sign_even = (sign_all & 0x08080808) at bit positions 3
    uint32_t sign_even = q_even & 0x08080808u;
    uint32_t sign_odd  = q_odd  & 0x08080808u;

    // Lookups
    uint32_t pos_even, pos_odd, neg_even, neg_odd;
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(pos_even) : "v"(mag_src1), "v"(mag_src2), "v"(idx_even));
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(neg_even) : "v"(neg_src1), "v"(neg_src2), "v"(idx_even));
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(pos_odd)  : "v"(mag_src1), "v"(mag_src2), "v"(idx_odd));
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(neg_odd)  : "v"(neg_src1), "v"(neg_src2), "v"(idx_odd));

    // Merge using perm
    uint32_t merge_even = 0x03020100u | (sign_even >> 1);
    uint32_t merge_odd  = 0x03020100u | (sign_odd >> 1);

    uint32_t res_even, res_odd;
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(res_even) : "v"(neg_even), "v"(pos_even), "v"(merge_even));
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(res_odd)  : "v"(neg_odd),  "v"(pos_odd),  "v"(merge_odd));

    *reinterpret_cast<uint32_t*>(result_even) = res_even;
    *reinterpret_cast<uint32_t*>(result_odd)  = res_odd;
}

//=============================================================================
// NEW: Optimization 4 - Single unified lookup with 8-entry tables
//=============================================================================

__device__ __forceinline__
void opt4_unified_single_lookup(uint32_t q, int8_t* result_even, int8_t* result_odd) {
    // Key insight: v_perm can select from 8 bytes using indices 0-7
    // We have exactly 8 unique magnitudes: {0,1,2,3,4,6,8,12}
    // We have exactly 8 unique negatives: {0,-1,-2,-3,-4,-6,-8,-12}
    //
    // Pack positive values in src1|src2 = {0,1,2,3,4,6,8,12}
    // Pack negative values in another src1|src2 pair
    // Use a single perm for positive, single perm for negative
    // Then select based on sign bit

    // Actually this is the same as opt1b... let me think differently

    // NEW APPROACH: Interleaved positive/negative table
    // If we pack: {0, 0, 1, -1, 2, -2, 3, -3} we could use adjusted indices
    // But this doesn't help with the 8-byte limit

    // BETTER IDEA: Use the fact that we're doing 4 perms already
    // Can we reduce further by processing more bytes in parallel?

    // For now, implement the cleanest 4-perm version
    constexpr uint32_t pos_lo = 0x03020100u;  // Bytes 0-3: {0,1,2,3}
    constexpr uint32_t pos_hi = 0x0C080604u;  // Bytes 4-7: {4,6,8,12}
    constexpr uint32_t neg_lo = 0xFDFEFF00u;  // Bytes 0-3: {0,-1,-2,-3}
    constexpr uint32_t neg_hi = 0xF4F8FAFCu;  // Bytes 4-7: {-4,-6,-8,-12}

    uint32_t q_even = q & 0x0F0F0F0Fu;
    uint32_t q_odd  = (q >> 4) & 0x0F0F0F0Fu;
    uint32_t idx_even = q_even & 0x07070707u;
    uint32_t idx_odd  = q_odd & 0x07070707u;

    uint32_t pos_e, neg_e, pos_o, neg_o;
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(pos_e) : "v"(pos_hi), "v"(pos_lo), "v"(idx_even));
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(neg_e) : "v"(neg_hi), "v"(neg_lo), "v"(idx_even));
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(pos_o) : "v"(pos_hi), "v"(pos_lo), "v"(idx_odd));
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(neg_o) : "v"(neg_hi), "v"(neg_lo), "v"(idx_odd));

    // Faster merge: Use v_cndmask_b32 with SDWA byte selection
    // But v_cndmask works at 32-bit level with VCC...
    // We need per-byte condition

    // Use perm-based merge
    uint32_t mask_e = 0x03020100u | ((q_even & 0x08080808u) >> 1);
    uint32_t mask_o = 0x03020100u | ((q_odd & 0x08080808u) >> 1);

    uint32_t res_e, res_o;
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(res_e) : "v"(neg_e), "v"(pos_e), "v"(mask_e));
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(res_o) : "v"(neg_o), "v"(pos_o), "v"(mask_o));

    *reinterpret_cast<uint32_t*>(result_even) = res_e;
    *reinterpret_cast<uint32_t*>(result_odd)  = res_o;
}

//=============================================================================
// NEW: Optimization 5 - Hybrid compute for small magnitudes
//=============================================================================

__device__ __forceinline__
void opt5_hybrid_compute(uint32_t q, int8_t* result_even, int8_t* result_odd) {
    // Insight: Indices 0-3 have magnitude = index (identity mapping!)
    // Only indices 4-7 need lookup for {4, 6, 8, 12}
    //
    // Split processing:
    // - If bit2=0: magnitude = idx (no lookup needed)
    // - If bit2=1: magnitude = lookup[idx & 0x03] from {4,6,8,12}
    //
    // Then apply sign

    constexpr uint32_t high_mag_tbl = 0x0C080604u;  // {4,6,8,12}
    constexpr uint32_t high_neg_tbl = 0xF4F8FAFCu;  // {-4,-6,-8,-12}

    uint32_t q_even = q & 0x0F0F0F0Fu;
    uint32_t q_odd  = (q >> 4) & 0x0F0F0F0Fu;

    // Low magnitude (identity for 0-3)
    uint32_t low_mag_even = q_even & 0x03030303u;
    uint32_t low_mag_odd  = q_odd  & 0x03030303u;

    // High magnitude (lookup for 4-7)
    uint32_t high_idx_even = q_even & 0x03030303u;  // lower 2 bits
    uint32_t high_idx_odd  = q_odd  & 0x03030303u;
    uint32_t high_mag_even, high_mag_odd;
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(high_mag_even) : "v"(high_mag_tbl), "v"(high_mag_tbl), "v"(high_idx_even));
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(high_mag_odd)  : "v"(high_mag_tbl), "v"(high_mag_tbl), "v"(high_idx_odd));

    // Select based on bit2
    uint32_t bit2_even = (q_even >> 2) & 0x01010101u;
    uint32_t bit2_odd  = (q_odd >> 2) & 0x01010101u;

    // Need to select between low_mag and high_mag per byte
    // Use perm with selector from bit2
    // If bit2=0: select from low (bytes 0-3), if bit2=1: select from high (bytes 4-7)
    // Pack low in bytes 0-3, high in bytes 4-7
    // selector = 0x03020100 | (bit2 << 2) = 0,1,2,3 or 4,5,6,7 per byte
    uint32_t sel_even = 0x03020100u | (bit2_even << 2);
    uint32_t sel_odd  = 0x03020100u | (bit2_odd << 2);

    // Combine low and high mags into single register for perm
    // This requires packing which adds overhead... not better

    // Alternative: Use AND/OR masking
    // mask = bit2 ? 0xFF : 0x00 per byte
    // result = (low & ~mask) | (high & mask)
    // Need to expand bit2 to full byte mask first

    // Expand bit2: bit2 * 0xFF
    // Use: perm with {0x00, 0xFF, ...} table
    constexpr uint32_t expand_tbl = 0x0000FF00u;  // byte0=0, byte1=0xFF
    uint32_t mask_even, mask_odd;
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(mask_even) : "v"(expand_tbl), "v"(expand_tbl), "v"(bit2_even));
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(mask_odd)  : "v"(expand_tbl), "v"(expand_tbl), "v"(bit2_odd));

    // Select magnitude: (low & ~mask) | (high & mask)
    uint32_t mag_even = (low_mag_even & ~mask_even) | (high_mag_even & mask_even);
    uint32_t mag_odd  = (low_mag_odd  & ~mask_odd)  | (high_mag_odd  & mask_odd);

    // Now handle sign (bit 3)
    uint32_t sign_even = q_even & 0x08080808u;
    uint32_t sign_odd  = q_odd  & 0x08080808u;

    // Need negative magnitudes too
    // low_neg = -(idx) for idx 0-3 = {0, -1, -2, -3} = {0x00, 0xFF, 0xFE, 0xFD}
    // Can we compute this? -x = ~x + 1
    // For 0-3: ~0x00=0xFF, ~0x01=0xFE, ~0x02=0xFD, ~0x03=0xFC
    // +1: 0x00, 0xFF, 0xFE, 0xFD ✓
    // BUT: adding 1 to packed bytes causes carry!
    // ~0x03030303 + 0x01010101 = 0xFCFCFCFC + 0x01010101 = 0xFDFDFDFD ✓ (no carry in this case!)

    // Actually for values 0-3, XOR + add works without carry because max = 0x03:
    // ~0x03 = 0xFC, +1 = 0xFD (no carry to bit 8)
    // BUT: if all bytes are different, we still might get carry...
    // ~0x00010203 + 0x01010101 = 0xFEFDFCFC + 0x01010101 = 0xFFFEFDFD (no carry since each <0xFF)
    // 0xFE + 1 = 0xFF (no carry), 0xFD + 1 = 0xFE, etc.

    // Safe per-byte negation for values 0-3:
    uint32_t low_neg_even = (~low_mag_even) + 0x01010101u;
    uint32_t low_neg_odd  = (~low_mag_odd)  + 0x01010101u;

    // For high magnitudes {4,6,8,12}, use lookup for negatives
    uint32_t high_neg_even, high_neg_odd;
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(high_neg_even) : "v"(high_neg_tbl), "v"(high_neg_tbl), "v"(high_idx_even));
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(high_neg_odd)  : "v"(high_neg_tbl), "v"(high_neg_tbl), "v"(high_idx_odd));

    // Select negative magnitude based on bit2
    uint32_t neg_mag_even = (low_neg_even & ~mask_even) | (high_neg_even & mask_even);
    uint32_t neg_mag_odd  = (low_neg_odd  & ~mask_odd)  | (high_neg_odd  & mask_odd);

    // Finally select between positive and negative based on sign
    uint32_t sign_sel_even = 0x03020100u | (sign_even >> 1);
    uint32_t sign_sel_odd  = 0x03020100u | (sign_odd >> 1);

    uint32_t res_even, res_odd;
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(res_even) : "v"(neg_mag_even), "v"(mag_even), "v"(sign_sel_even));
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(res_odd)  : "v"(neg_mag_odd),  "v"(mag_odd),  "v"(sign_sel_odd));

    *reinterpret_cast<uint32_t*>(result_even) = res_even;
    *reinterpret_cast<uint32_t*>(result_odd)  = res_odd;
}

//=============================================================================
// NEW: Optimization 6 - Minimal 4-perm with optimized merging
//=============================================================================

__device__ __forceinline__
void opt6_minimal_4perm(uint32_t q, int8_t* result_even, int8_t* result_odd) {
    // Observation: The merge mask generation is the overhead
    // Current: mask = 0x03020100 | ((q & 0x08) >> 1)
    // This takes 2 ops (AND + shift + OR)

    // Can we encode the merge condition directly?
    // v_perm indices: 0-3 select from src2, 4-7 select from src1
    // So we need: if sign=0, use 0,1,2,3; if sign=1, use 4,5,6,7

    // Current encoding: base = 0,1,2,3; offset by 4 if sign set
    // sign_offset = ((q >> 3) & 1) << 2 = (q >> 1) & 4
    // mask[byte] = byte_idx + sign_offset

    // Precompute: mask = (q >> 1) & 0x04040404 | 0x03020100

    constexpr uint32_t pos_lo = 0x03020100u;
    constexpr uint32_t pos_hi = 0x0C080604u;
    constexpr uint32_t neg_lo = 0xFDFEFF00u;
    constexpr uint32_t neg_hi = 0xF4F8FAFCu;

    uint32_t q_even = q & 0x0F0F0F0Fu;
    uint32_t q_odd  = (q >> 4) & 0x0F0F0F0Fu;
    uint32_t idx_even = q_even & 0x07070707u;
    uint32_t idx_odd  = q_odd & 0x07070707u;

    // Optimized mask: shift sign bit (3) right by 1 to position (2) = offset 4
    // Then OR with base 0x03020100
    uint32_t mask_e = 0x03020100u | ((q_even >> 1) & 0x04040404u);
    uint32_t mask_o = 0x03020100u | ((q_odd  >> 1) & 0x04040404u);

    uint32_t pos_e, neg_e, pos_o, neg_o;
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(pos_e) : "v"(pos_hi), "v"(pos_lo), "v"(idx_even));
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(neg_e) : "v"(neg_hi), "v"(neg_lo), "v"(idx_even));
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(pos_o) : "v"(pos_hi), "v"(pos_lo), "v"(idx_odd));
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(neg_o) : "v"(neg_hi), "v"(neg_lo), "v"(idx_odd));

    uint32_t res_e, res_o;
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(res_e) : "v"(neg_e), "v"(pos_e), "v"(mask_e));
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(res_o) : "v"(neg_o), "v"(pos_o), "v"(mask_o));

    *reinterpret_cast<uint32_t*>(result_even) = res_e;
    *reinterpret_cast<uint32_t*>(result_odd)  = res_o;
}

//=============================================================================
// Benchmark kernels
//=============================================================================

__global__ void benchmark_baseline(uint32_t* __restrict__ out, const uint32_t* __restrict__ in, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    uint32_t q = in[idx];
    int8_t result_even[4], result_odd[4];

    #pragma unroll
    for (int i = 0; i < 100; i++) {
        baseline_6perm(q, result_even, result_odd);
        q ^= *reinterpret_cast<uint32_t*>(result_even);  // Prevent optimization
    }

    out[idx*2]   = *reinterpret_cast<uint32_t*>(result_even);
    out[idx*2+1] = *reinterpret_cast<uint32_t*>(result_odd);
}

__global__ void benchmark_opt1a(uint32_t* __restrict__ out, const uint32_t* __restrict__ in, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    uint32_t q = in[idx];
    int8_t result_even[4], result_odd[4];

    #pragma unroll
    for (int i = 0; i < 100; i++) {
        opt1a_magnitude_sign_bfi(q, result_even, result_odd);
        q ^= *reinterpret_cast<uint32_t*>(result_even);
    }

    out[idx*2]   = *reinterpret_cast<uint32_t*>(result_even);
    out[idx*2+1] = *reinterpret_cast<uint32_t*>(result_odd);
}

__global__ void benchmark_opt1b(uint32_t* __restrict__ out, const uint32_t* __restrict__ in, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    uint32_t q = in[idx];
    int8_t result_even[4], result_odd[4];

    #pragma unroll
    for (int i = 0; i < 100; i++) {
        opt1b_magnitude_xor_sign(q, result_even, result_odd);
        q ^= *reinterpret_cast<uint32_t*>(result_even);
    }

    out[idx*2]   = *reinterpret_cast<uint32_t*>(result_even);
    out[idx*2+1] = *reinterpret_cast<uint32_t*>(result_odd);
}

__global__ void benchmark_opt2(uint32_t* __restrict__ out, const uint32_t* __restrict__ in, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    uint32_t q = in[idx];
    int8_t result_even[4], result_odd[4];

    #pragma unroll
    for (int i = 0; i < 100; i++) {
        opt2_oob_zeroing(q, result_even, result_odd);
        q ^= *reinterpret_cast<uint32_t*>(result_even);
    }

    out[idx*2]   = *reinterpret_cast<uint32_t*>(result_even);
    out[idx*2+1] = *reinterpret_cast<uint32_t*>(result_odd);
}

__global__ void benchmark_opt6(uint32_t* __restrict__ out, const uint32_t* __restrict__ in, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    uint32_t q = in[idx];
    int8_t result_even[4], result_odd[4];

    #pragma unroll
    for (int i = 0; i < 100; i++) {
        opt6_minimal_4perm(q, result_even, result_odd);
        q ^= *reinterpret_cast<uint32_t*>(result_even);
    }

    out[idx*2]   = *reinterpret_cast<uint32_t*>(result_even);
    out[idx*2+1] = *reinterpret_cast<uint32_t*>(result_odd);
}

//=============================================================================
// Correctness verification kernel
//=============================================================================

__global__ void verify_correctness(uint32_t* results, int* errors) {
    // Test all 65536 possible 16-bit input patterns
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 65536) return;

    uint32_t q = idx | (idx << 16);  // Replicate pattern

    int8_t baseline_even[4], baseline_odd[4];
    int8_t opt1b_even[4], opt1b_odd[4];
    int8_t opt6_even[4], opt6_odd[4];

    baseline_6perm(q, baseline_even, baseline_odd);
    opt1b_magnitude_xor_sign(q, opt1b_even, opt1b_odd);
    opt6_minimal_4perm(q, opt6_even, opt6_odd);

    uint32_t base_e = *reinterpret_cast<uint32_t*>(baseline_even);
    uint32_t base_o = *reinterpret_cast<uint32_t*>(baseline_odd);
    uint32_t o1b_e = *reinterpret_cast<uint32_t*>(opt1b_even);
    uint32_t o1b_o = *reinterpret_cast<uint32_t*>(opt1b_odd);
    uint32_t o6_e = *reinterpret_cast<uint32_t*>(opt6_even);
    uint32_t o6_o = *reinterpret_cast<uint32_t*>(opt6_odd);

    if (base_e != o1b_e || base_o != o1b_o) {
        atomicAdd(errors, 1);
        if (atomicAdd(errors + 1, 1) < 10) {
            results[idx * 6 + 0] = q;
            results[idx * 6 + 1] = base_e;
            results[idx * 6 + 2] = o1b_e;
            results[idx * 6 + 3] = base_o;
            results[idx * 6 + 4] = o1b_o;
            results[idx * 6 + 5] = 0xDEAD0001;
        }
    }
    if (base_e != o6_e || base_o != o6_o) {
        atomicAdd(errors + 2, 1);
        if (atomicAdd(errors + 3, 1) < 10) {
            results[idx * 6 + 0] = q;
            results[idx * 6 + 1] = base_e;
            results[idx * 6 + 2] = o6_e;
            results[idx * 6 + 3] = base_o;
            results[idx * 6 + 4] = o6_o;
            results[idx * 6 + 5] = 0xDEAD0006;
        }
    }
}

//=============================================================================
// Main
//=============================================================================

int main() {
    printf("MXFP4 Byte Operation Optimization Tests for GFX906\n");
    printf("===================================================\n\n");

    // Verify correctness first
    printf("Phase 1: Correctness Verification\n");
    printf("---------------------------------\n");

    uint32_t* d_results;
    int* d_errors;
    hipMalloc(&d_results, 65536 * 6 * sizeof(uint32_t));
    hipMalloc(&d_errors, 4 * sizeof(int));
    hipMemset(d_errors, 0, 4 * sizeof(int));

    verify_correctness<<<256, 256>>>(d_results, d_errors);
    hipDeviceSynchronize();

    int h_errors[4];
    hipMemcpy(h_errors, d_errors, 4 * sizeof(int), hipMemcpyDeviceToHost);

    printf("opt1b errors: %d\n", h_errors[0]);
    printf("opt6 errors:  %d\n", h_errors[2]);

    if (h_errors[0] > 0 || h_errors[2] > 0) {
        printf("\n⚠️  Some optimizations produce incorrect results!\n");
    } else {
        printf("\n✓ All optimizations produce correct results!\n");
    }

    hipFree(d_results);
    hipFree(d_errors);

    // Performance benchmark
    printf("\nPhase 2: Performance Benchmark\n");
    printf("------------------------------\n");

    const int N = 1024 * 1024;  // 1M elements
    const int WARMUP = 5;
    const int RUNS = 20;

    uint32_t* d_in;
    uint32_t* d_out;
    hipMalloc(&d_in, N * sizeof(uint32_t));
    hipMalloc(&d_out, N * 2 * sizeof(uint32_t));

    // Initialize with random data
    uint32_t* h_in = new uint32_t[N];
    for (int i = 0; i < N; i++) {
        h_in[i] = rand();
    }
    hipMemcpy(d_in, h_in, N * sizeof(uint32_t), hipMemcpyHostToDevice);
    delete[] h_in;

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    auto benchmark = [&](const char* name, auto kernel) {
        // Warmup
        for (int i = 0; i < WARMUP; i++) {
            kernel<<<gridSize, blockSize>>>(d_out, d_in, N);
        }
        hipDeviceSynchronize();

        // Timed runs
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < RUNS; i++) {
            kernel<<<gridSize, blockSize>>>(d_out, d_in, N);
        }
        hipDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();

        double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
        double avg_ms = elapsed_ms / RUNS;
        double ops_per_sec = (N * 100.0) / (avg_ms / 1000.0);  // 100 iterations in kernel

        printf("%-20s: %.3f ms avg, %.2f B ops/sec\n", name, avg_ms, ops_per_sec / 1e9);
    };

    benchmark("Baseline (6 perm)", benchmark_baseline);
    benchmark("Opt1a (BFI)", benchmark_opt1a);
    benchmark("Opt1b (4+2 perm)", benchmark_opt1b);
    benchmark("Opt2 (OOB zero)", benchmark_opt2);
    benchmark("Opt6 (4 perm min)", benchmark_opt6);

    hipFree(d_in);
    hipFree(d_out);

    printf("\nDone.\n");
    return 0;
}
