// Test kernel for MXFP4 3 v_perm lookup exploration
// Goal: Find a combination that uses only 3 v_perm + cheap ops

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>

// Reference lookup table
__device__ __constant__ int8_t kvalues_mxfp4[16] = {
    0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12
};

// ============================================================================
// REFERENCE: Original 6 v_perm implementation
// ============================================================================
__device__ __forceinline__ int2 lookup_reference(uint32_t q4) {
    const uint32_t *values = (const uint32_t *)kvalues_mxfp4;

    const uint32_t q_even = q4;
    const uint32_t q_odd  = (q4 >> 4);

    // 4 v_perm for table lookups
    uint32_t v_even_low  = __builtin_amdgcn_perm(values[1], values[0], q_even & 0x07070707);
    uint32_t v_odd_low   = __builtin_amdgcn_perm(values[1], values[0], q_odd  & 0x07070707);
    uint32_t v_even_high = __builtin_amdgcn_perm(values[3], values[2], q_even & 0x07070707);
    uint32_t v_odd_high  = __builtin_amdgcn_perm(values[3], values[2], q_odd  & 0x07070707);

    // 2 v_perm for selection
    uint32_t mask_even = 0x03020100 | ((q_even & 0x08080808) >> 1);
    uint32_t mask_odd  = 0x03020100 | ((q_odd  & 0x08080808) >> 1);
    uint32_t res_x = __builtin_amdgcn_perm(v_even_high, v_even_low, mask_even);
    uint32_t res_y = __builtin_amdgcn_perm(v_odd_high,  v_odd_low,  mask_odd);

    return make_int2(res_x, res_y);
}

// ============================================================================
// APPROACH 1: 2 v_perm for magnitude + v_cndmask per-word selection
// Uses the fact that we can compute neg = ~mag + 0x01010101 IF we handle
// the carry issue with masking
// ============================================================================
__device__ __forceinline__ int2 lookup_approach1(uint32_t q4) {
    // Magnitude-only table (8 entries)
    const uint32_t mag_lo = 0x03020100;  // {0, 1, 2, 3}
    const uint32_t mag_hi = 0x0c080604;  // {4, 6, 8, 12}

    const uint32_t q_even = q4;
    const uint32_t q_odd  = q4 >> 4;
    const uint32_t mag_idx_even = q_even & 0x07070707;
    const uint32_t mag_idx_odd  = q_odd  & 0x07070707;

    // 2 v_perm for magnitude lookup
    uint32_t mag_even = __builtin_amdgcn_perm(mag_hi, mag_lo, mag_idx_even);
    uint32_t mag_odd  = __builtin_amdgcn_perm(mag_hi, mag_lo, mag_idx_odd);

    // Extract sign bits and expand to byte masks
    // sign_bits: 0x01 where negative, 0x00 where positive
    uint32_t sign_even = (q_even >> 3) & 0x01010101;
    uint32_t sign_odd  = (q_odd  >> 3) & 0x01010101;

    // Compute negative: -mag = ~mag + 1 per byte
    // But we have carry issues... try a different approach:
    // neg = (mag ^ 0xFF) + 1 = 0x100 - mag
    // For mag <= 12, this fits in a byte (except mag=0 gives 0x100 -> 0x00 with carry)

    // Use SDWA trick: v_sub per-byte
    // Since we don't have direct per-byte sub, let's try packed approach

    // Actually, let's use the BFI approach but try to reduce v_perm count
    // The sign expansion still needs 1 v_perm per word = 2 total

    // Alternative: Use arithmetic to expand sign
    // If sign_bits = 0x01010101 for all negative:
    // sign_mask = -sign_bits would give 0xFEFEFEFF (wrong due to borrow)
    // What about: sign_mask = (sign_bits << 8) - sign_bits?
    // That shifts the whole word, not per-byte.

    // Let's try: sign_mask = sign_bits * 255
    // We can compute this as: (sign_bits << 8) - sign_bits for 16-bit words
    // Then combine... this is getting complex

    // For now, use v_perm for sign expansion (adds 2 v_perm = 4 total)
    const uint32_t expand = 0x0000FF00;
    uint32_t sign_mask_even = __builtin_amdgcn_perm(expand, expand, sign_even);
    uint32_t sign_mask_odd  = __builtin_amdgcn_perm(expand, expand, sign_odd);

    // Compute negative with saturating-style approach
    // neg = 0 - mag, but per-byte
    // Since mag <= 12, we can use: neg = (0x01010100 - mag) ^ 0x01010101 + carry_fix
    // Actually let's just compute it directly

    // neg = (~mag + 0x01010101) & byte_mask where mag != 0
    // For mag = 0: we want neg = 0, but ~0 + 1 = 0x100 which carries

    // Hack: compute both and select
    uint32_t neg_even = ((~mag_even) + 0x01010101);
    uint32_t neg_odd  = ((~mag_odd)  + 0x01010101);

    // Fix carry corruption: where mag was 0, neg should be 0
    // zero_mask = (mag == 0) ? 0xFF : 0x00 per byte
    // This requires comparison... more ops

    // Let's use BFI to select between pos and neg
    uint32_t res_x, res_y;
    asm volatile("v_bfi_b32 %0, %1, %2, %3" : "=v"(res_x) : "v"(sign_mask_even), "v"(neg_even), "v"(mag_even));
    asm volatile("v_bfi_b32 %0, %1, %2, %3" : "=v"(res_y) : "v"(sign_mask_odd),  "v"(neg_odd),  "v"(mag_odd));

    return make_int2(res_x, res_y);
}

// ============================================================================
// APPROACH 2: 3 v_perm with shared sign expansion
// Idea: Process both even and odd with a single sign expansion v_perm
// ============================================================================
__device__ __forceinline__ int2 lookup_approach2(uint32_t q4) {
    // Positive and negative tables
    const uint32_t pos_lo = 0x03020100;  // {0, 1, 2, 3}
    const uint32_t pos_hi = 0x0c080604;  // {4, 6, 8, 12}
    const uint32_t neg_lo = 0xfdfeff00;  // {0, -1, -2, -3}
    const uint32_t neg_hi = 0xf4f8fafc;  // {-4, -6, -8, -12}

    const uint32_t q_even = q4;
    const uint32_t q_odd  = q4 >> 4;
    const uint32_t mag_idx_even = q_even & 0x07070707;
    const uint32_t mag_idx_odd  = q_odd  & 0x07070707;

    // Extract sign bits (bit 3 of each nibble)
    uint32_t sign_even = (q_even >> 3) & 0x01010101;
    uint32_t sign_odd  = (q_odd  >> 3) & 0x01010101;

    // Pack sign bits into bytes 0-3 for even, 4-7 for odd
    // Then use single v_perm to expand all 8 signs at once?
    // Problem: v_perm outputs 4 bytes, we need 8 bytes of sign masks

    // Actually, let's try a different trick:
    // Use DPP to share work between lanes?
    // Or use the sign bits to directly index pos/neg tables

    // Alternative: Interleaved table layout
    // Table: {pos[0], neg[0], pos[1], neg[1], ...}
    // Index: 2*magnitude + sign
    // But this changes the index computation...

    // Let's try: compute index = (magnitude << 1) | sign
    // For 8 magnitudes and 2 signs = 16 entries
    // But we'd need a new table layout

    // Interleaved table:
    // {0, 0, 1, -1, 2, -2, 3, -3, 4, -4, 6, -6, 8, -8, 12, -12}
    // Wait, that's 16 entries but indices don't match cleanly

    // Let me try the original approach with v_perm reduction

    // 2 v_perm for positive magnitudes
    uint32_t pos_even = __builtin_amdgcn_perm(pos_hi, pos_lo, mag_idx_even);
    uint32_t pos_odd  = __builtin_amdgcn_perm(pos_hi, pos_lo, mag_idx_odd);

    // 2 v_perm for negative magnitudes
    uint32_t neg_even = __builtin_amdgcn_perm(neg_hi, neg_lo, mag_idx_even);
    uint32_t neg_odd  = __builtin_amdgcn_perm(neg_hi, neg_lo, mag_idx_odd);

    // Can we use a single v_perm for sign expansion of BOTH even and odd?
    // We have sign_even and sign_odd, each with 4 bytes of 0/1 values
    // We need to expand each byte to 0x00 or 0xFF

    // Idea: Pack even signs in low 16 bits, odd signs in high 16 bits
    // Then use v_perm with special selector pattern
    // But v_perm selector is 32-bit with 4 byte selectors, outputting 4 bytes
    // We need 8 output bytes (2 x 32-bit words)

    // This doesn't reduce v_perm count for sign expansion

    // Let's try arithmetic sign expansion instead of v_perm:
    // sign_mask = 0 - sign_bits (for 0->0x00, 1->0xFF per byte)
    // Problem: borrow propagation

    // Workaround: Use left shift and subtract within 16-bit halves
    // sign16_lo = sign_even & 0x00FF00FF
    // sign16_hi = (sign_even >> 8) & 0x00FF00FF
    // mask16_lo = (sign16_lo * 255) within 16-bit
    // = sign16_lo * 256 - sign16_lo
    // = (sign16_lo << 8) - sign16_lo  (this overflows 16-bit for sign=1)

    // sign16_lo = 0x00010001 gives:
    // (0x00010001 << 8) = 0x01000100
    // 0x01000100 - 0x00010001 = 0x00FF00FF (!)
    // This works for the low byte of each 16-bit half!

    // But what about high bytes?
    // sign16_hi = 0x00010001 (if bytes 1 and 3 have sign=1)
    // Same calculation gives 0x00FF00FF for the low halves
    // But we need these in the high byte positions

    // Full approach:
    // sign_lo = sign_even & 0x00FF00FF  // bytes 0 and 2
    // sign_hi = (sign_even >> 8) & 0x00FF00FF  // bytes 1 and 3
    // mask_lo = (sign_lo << 8) - sign_lo  // 0x00 or 0xFF in positions 0, 2
    // mask_hi = (sign_hi << 8) - sign_hi  // 0x00 or 0xFF for positions 1, 3
    // sign_mask = (mask_lo & 0x00FF00FF) | ((mask_hi & 0x00FF00FF) << 8)

    // That's 8 ops per sign mask, 16 total. Way worse than 2 v_perm!

    // Back to 2 v_perm for sign expansion
    const uint32_t expand = 0x0000FF00;
    uint32_t mask_even = __builtin_amdgcn_perm(expand, expand, sign_even);
    uint32_t mask_odd  = __builtin_amdgcn_perm(expand, expand, sign_odd);

    // 2 BFI for final selection
    uint32_t res_x, res_y;
    asm volatile("v_bfi_b32 %0, %1, %2, %3" : "=v"(res_x) : "v"(mask_even), "v"(neg_even), "v"(pos_even));
    asm volatile("v_bfi_b32 %0, %1, %2, %3" : "=v"(res_y) : "v"(mask_odd),  "v"(neg_odd),  "v"(pos_odd));

    return make_int2(res_x, res_y);  // Total: 6 v_perm + 2 BFI = 8 ops
}

// ============================================================================
// APPROACH 3: Use DPP for lane-level table distribution
// Idea: Distribute table across 4 lanes, use DPP quad_perm to gather
// ============================================================================
__device__ __forceinline__ int2 lookup_approach3(uint32_t q4, int lane_id) {
    // This approach requires coordination across lanes
    // Each lane in a quad holds part of the table:
    // Lane 0: entries 0-3
    // Lane 1: entries 4-7
    // Lane 2: entries 8-11
    // Lane 3: entries 12-15

    // But DPP quad_perm has a fixed permutation pattern, not data-dependent
    // So we can't dynamically select which lane to read from based on q4

    // This approach won't work directly. Would need multiple DPP ops + selection.

    // Fall back to reference for now
    return lookup_reference(q4);
}

// ============================================================================
// APPROACH 4: Magnitude lookup + XOR/arithmetic sign (avoiding carry issue)
// Key insight: For mag > 0, we can safely negate. For mag = 0, result is 0 anyway.
// ============================================================================
__device__ __forceinline__ int2 lookup_approach4(uint32_t q4) {
    // Magnitude table
    const uint32_t mag_lo = 0x03020100;  // {0, 1, 2, 3}
    const uint32_t mag_hi = 0x0c080604;  // {4, 6, 8, 12}

    const uint32_t q_even = q4;
    const uint32_t q_odd  = q4 >> 4;
    const uint32_t mag_idx_even = q_even & 0x07070707;
    const uint32_t mag_idx_odd  = q_odd  & 0x07070707;

    // 2 v_perm for magnitude lookup
    uint32_t mag_even = __builtin_amdgcn_perm(mag_hi, mag_lo, mag_idx_even);
    uint32_t mag_odd  = __builtin_amdgcn_perm(mag_hi, mag_lo, mag_idx_odd);

    // Extract sign bits
    uint32_t sign_even = (q_even >> 3) & 0x01010101;
    uint32_t sign_odd  = (q_odd  >> 3) & 0x01010101;

    // Two's complement: -x = ~x + 1 = (x ^ 0xFF) + 1
    // For per-byte: neg = (mag ^ sign_mask) + sign_bits
    // where sign_mask = 0xFF per negative byte, 0x00 per positive byte

    // Sign expansion (this is the expensive part)
    // Option A: v_perm (2 ops)
    // Option B: Arithmetic trick

    // Arithmetic expansion attempt:
    // mask = -sign (would work if per-byte)
    // 0x00000000 - 0x00000001 = 0xFFFFFFFF (works for single bit)
    // 0x00000000 - 0x01010101 = 0xFEFEFEFF (wrong due to borrow chain)

    // Alternative: left-shift and compare
    // sign_shifted = sign << 7  // Move sign bit to MSB of each byte
    // sign_extended = (int32_t)sign_shifted >> 31  // Arithmetic right shift
    // But this extends the MSB of the word, not per-byte

    // Per-byte arithmetic shift doesn't exist directly

    // What about using multiply?
    // sign * 0xFF per byte... but no per-byte multiply

    // v_pk_mul can do packed 16-bit multiply
    // If we treat pairs of bytes as 16-bit values:
    // sign16 = sign & 0x00010001 (bytes 0 and 2 as 16-bit)
    // mask16 = sign16 * 0x00FF00FF per 16-bit? No, that's still wrong.

    // What about: mask = (sign << 8) - sign within 16-bit halves?
    // For sign16 = 0x0001: (0x0001 << 8) - 0x0001 = 0x0100 - 0x0001 = 0x00FF
    // That gives the mask for byte 0!

    // But we need byte 1 as well. And we need to interleave the results.

    // Let's try this approach:
    // Step 1: Separate even and odd bytes of sign
    uint32_t sign_even_b02 = sign_even & 0x00FF00FF;  // bytes 0, 2 in 16-bit slots
    uint32_t sign_even_b13 = (sign_even >> 8) & 0x00FF00FF;  // bytes 1, 3 in 16-bit slots
    uint32_t sign_odd_b02 = sign_odd & 0x00FF00FF;
    uint32_t sign_odd_b13 = (sign_odd >> 8) & 0x00FF00FF;

    // Step 2: Compute mask using 16-bit arithmetic (no cross-lane carry in pk_sub)
    // mask = (sign << 8) - sign, but using packed 16-bit sub
    uint32_t tmp_even_02, tmp_even_13, tmp_odd_02, tmp_odd_13;

    // Using v_pk_sub_u16: dst = src0 - src1 (packed 16-bit)
    asm volatile("v_pk_sub_u16 %0, %1, %2" : "=v"(tmp_even_02) : "v"(sign_even_b02 << 8), "v"(sign_even_b02));
    asm volatile("v_pk_sub_u16 %0, %1, %2" : "=v"(tmp_even_13) : "v"(sign_even_b13 << 8), "v"(sign_even_b13));
    asm volatile("v_pk_sub_u16 %0, %1, %2" : "=v"(tmp_odd_02) : "v"(sign_odd_b02 << 8), "v"(sign_odd_b02));
    asm volatile("v_pk_sub_u16 %0, %1, %2" : "=v"(tmp_odd_13) : "v"(sign_odd_b13 << 8), "v"(sign_odd_b13));

    // Step 3: Recombine bytes
    // tmp_*_02 has mask for bytes 0, 2 in the low byte of each 16-bit half
    // tmp_*_13 has mask for bytes 1, 3 in the low byte of each 16-bit half
    uint32_t mask_even = (tmp_even_02 & 0x00FF00FF) | ((tmp_even_13 & 0x00FF00FF) << 8);
    uint32_t mask_odd  = (tmp_odd_02  & 0x00FF00FF) | ((tmp_odd_13  & 0x00FF00FF) << 8);

    // Now apply sign: result = (mag ^ mask) + sign_bits
    uint32_t res_x = (mag_even ^ mask_even) + sign_even;
    uint32_t res_y = (mag_odd  ^ mask_odd)  + sign_odd;

    return make_int2(res_x, res_y);
}

// ============================================================================
// APPROACH 5: Single interleaved table (16 entries in special order)
// ============================================================================
__device__ __forceinline__ int2 lookup_approach5(uint32_t q4) {
    // Interleaved table: even indices = positive, odd indices = negative
    // table[2*mag] = +mag_value, table[2*mag+1] = -mag_value
    // New index = 2 * magnitude + sign
    //           = 2 * (old_index & 7) + (old_index >> 3)
    //           = ((old_index & 7) << 1) | (old_index >> 3)

    // Interleaved table:
    // idx 0: mag=0, sign=0 -> 0
    // idx 1: mag=0, sign=1 -> 0
    // idx 2: mag=1, sign=0 -> 1
    // idx 3: mag=1, sign=1 -> -1
    // idx 4: mag=2, sign=0 -> 2
    // idx 5: mag=2, sign=1 -> -2
    // idx 6: mag=3, sign=0 -> 3
    // idx 7: mag=3, sign=1 -> -3
    // idx 8: mag=4, sign=0 -> 4
    // idx 9: mag=4, sign=1 -> -4
    // idx 10: mag=5, sign=0 -> 6
    // idx 11: mag=5, sign=1 -> -6
    // idx 12: mag=6, sign=0 -> 8
    // idx 13: mag=6, sign=1 -> -8
    // idx 14: mag=7, sign=0 -> 12
    // idx 15: mag=7, sign=1 -> -12

    const uint32_t table_0 = 0xFFFF0100;  // idx 0-3: {0, 0, 1, -1}
    const uint32_t table_1 = 0xFDFE0302;  // idx 4-7: {2, 3, -2, -3} -- wait, let me recalc

    // Bytes in little-endian order:
    // table_0 bytes: [0]=0, [1]=0, [2]=1, [3]=-1(0xFF)
    // = 0x00 | (0x00 << 8) | (0x01 << 16) | (0xFF << 24) = 0xFF010000
    const uint32_t tbl_0 = 0xFF010000;  // {0, 0, 1, -1}
    const uint32_t tbl_1 = 0xFD020302;  // wait, {2, 3, -2, -3} in little endian:
    // [0]=2, [1]=3, [2]=-2(0xFE), [3]=-3(0xFD)
    // = 0x02 | (0x03 << 8) | (0xFE << 16) | (0xFD << 24) = 0xFDFE0302
    const uint32_t tbl_1_correct = 0xFDFE0302;  // {2, 3, -2, -3}

    // Actually I need to be more careful. v_perm uses byte positions 0-7
    // where src0 provides bytes 0-3 and src1 provides bytes 4-7

    // Table layout for v_perm(src1, src0, sel):
    // src0: bytes 0-3 = table indices 0-3
    // src1: bytes 0-3 = table indices 4-7
    // Selector byte n chooses byte (sel[n] & 7) from the 8-byte src1:src0 concatenation

    // For indices 0-7 (first 8 entries):
    // v_perm(tbl_1, tbl_0, sel) where sel bytes are 0-7

    // For indices 8-15 (next 8 entries):
    // Need another pair of registers

    // Let me properly set up the tables:
    // Interleaved values: {0, 0, 1, -1, 2, -2, 3, -3, 4, -4, 6, -6, 8, -8, 12, -12}
    // As bytes:           {0, 0, 1, FF, 2, FE, 3, FD, 4, FC, 6, FA, 8, F8, 12, F4}
    // In hex:             {00, 00, 01, FF, 02, FE, 03, FD, 04, FC, 06, FA, 08, F8, 0C, F4}

    // First 4 bytes (indices 0-3): 0x00, 0x00, 0x01, 0xFF -> 0xFF010000
    // Next 4 bytes (indices 4-7):  0x02, 0xFE, 0x03, 0xFD -> 0xFD03FE02
    // Next 4 bytes (indices 8-11): 0x04, 0xFC, 0x06, 0xFA -> 0xFA06FC04
    // Last 4 bytes (indices 12-15): 0x08, 0xF8, 0x0C, 0xF4 -> 0xF40CF808

    const uint32_t interleaved_lo_0 = 0xFF010000;  // indices 0-3
    const uint32_t interleaved_lo_1 = 0xFD03FE02;  // indices 4-7
    const uint32_t interleaved_hi_0 = 0xFA06FC04;  // indices 8-11
    const uint32_t interleaved_hi_1 = 0xF40CF808;  // indices 12-15

    // Compute new indices
    // new_idx = ((old_idx & 7) << 1) | (old_idx >> 3)
    // But we need to do this per nibble...

    // For each nibble in q4:
    // magnitude = nibble & 0x7
    // sign = (nibble >> 3) & 1
    // new_idx = (magnitude << 1) | sign

    // Per-nibble operation on 32-bit word:
    // Extract magnitudes: q & 0x07070707
    // Extract signs: (q >> 3) & 0x01010101
    // Combine: ((q & 0x07) << 1) | ((q >> 3) & 0x01) per nibble
    // = ((q << 1) & 0x0E) | ((q >> 3) & 0x01)
    // = (q & 0x07) * 2 + (q >> 3)

    // For packed nibbles:
    // magnitude_shifted = (q << 1) & 0x0E0E0E0E
    // signs = (q >> 3) & 0x01010101
    // new_idx = magnitude_shifted | signs

    // But wait, this gives us the new index for each nibble.
    // The new index is 4 bits (0-15), same as before.
    // So we still need 2 v_perm + selection for 16 entries!

    // The interleaving doesn't reduce v_perm count, it just changes the table layout.

    // Let me think differently...

    // Actually, with interleaving, the index computation becomes:
    // new_idx = (mag << 1) | sign
    //
    // For mag 0-3 (indices 0-7): use first pair of registers
    // For mag 4-7 (indices 8-15): use second pair of registers
    // Selection is based on mag[2] (bit 2 of magnitude)

    // But that's still the same complexity as before!
    // The selection is now on bit 2 of magnitude instead of bit 3 (sign).

    // OK, interleaving doesn't help. Let me abandon this approach.

    return lookup_reference(q4);
}

// ============================================================================
// APPROACH 6: v_perm with byte swap trick
// Observation: neg = -pos for non-zero entries
// If we can cheaply negate after a single lookup, we win
// ============================================================================
__device__ __forceinline__ int2 lookup_approach6(uint32_t q4) {
    // Table with both positive and pre-negated values
    // Positive: {0, 1, 2, 3, 4, 6, 8, 12}
    // Negative: {0, -1, -2, -3, -4, -6, -8, -12} = {0, 0xFF, 0xFE, 0xFD, 0xFC, 0xFA, 0xF8, 0xF4}

    // What if we store: positive in low nibble, negative in high nibble?
    // entry[i] = (neg[i] << 4) | pos[i]
    // Then use SDWA or shift to extract the right half

    // entry[0] = (0 << 4) | 0 = 0x00
    // entry[1] = (0xFF << 4) | 1 = 0xF1 -- wait, 0xFF << 4 = 0xFF0, doesn't fit in byte

    // This doesn't work for bytes. Would need 16-bit entries.

    // Alternative: Store pairs
    // reg contains {pos[0], neg[0], pos[1], neg[1]} as 4 bytes
    // Index by magnitude, then use byte offset based on sign

    // For magnitude m and sign s:
    // byte_offset = 2*m + s
    // This is exactly the interleaved approach from Approach 5

    return lookup_reference(q4);
}

// ============================================================================
// APPROACH 7: 3 v_perm using packed processing
// Process all 8 nibbles with 3 v_perm by clever packing
// ============================================================================
__device__ __forceinline__ int2 lookup_approach7(uint32_t q4) {
    // Idea: Instead of even/odd split, process all nibbles together
    // Pack 8 nibble indices into a 64-bit value, use 2 v_perm for low half + 2 for high half
    // Then... still need selection

    // Alternative: Use the selector bytes of v_perm more cleverly
    // The selector is 32 bits = 4 bytes, each byte selects one of 8 source bytes
    // If we compute the selector to include both magnitude lookup AND sign-based selection...

    // Original selector for positive magnitudes: mag_idx = q & 0x07070707
    // For negative magnitudes, we want: mag_idx | 0x08080808 (to select from high table)
    // No wait, that's wrong. High table is in bytes 4-7 of the source pair.

    // Actually, the selection v_perm uses mask = 0x03020100 | ((q & 0x08080808) >> 1)
    // This makes selector bytes: 0, 1, 2, 3 for low selection, 4, 5, 6, 7 for high selection

    // What if we combined the magnitude index and selection into one step?
    // Unified selector = base_select[magnitude] + 4 * sign
    // where base_select[m] = m (for m < 4), m (for m >= 4, from high table)

    // Hmm, this is getting confusing. Let me think about what v_perm actually does.

    // v_perm(src1, src0, sel) outputs 4 bytes:
    // out[i] = (sel[i] < 8) ? concat(src1, src0)[sel[i] & 7] : special_value

    // If we have:
    // src0 = table[0:3]  (4 bytes)
    // src1 = table[4:7]  (4 bytes)
    // sel[i] = table_index for nibble i

    // Then v_perm directly looks up 4 table entries (if indices < 8)

    // For 16-entry table:
    // First v_perm: src0 = table[0:3], src1 = table[4:7] -> accesses indices 0-7
    // Second v_perm: src0 = table[8:11], src1 = table[12:15] -> accesses indices 8-15
    // Selection v_perm: choose between first and second based on index >= 8

    // The selection uses index bit 3 (sign bit in our case) to choose.

    // Can we somehow encode the selection into the magnitude lookup?

    // What if we built the selector byte to include both magnitude and sign?
    // selector_byte = magnitude + 4 * sign (if sign chooses between low and high registers)

    // For q nibble = SMMM (S = sign, MMM = magnitude):
    // selector_byte = MMM + 4 * S = (S << 2) | MMM

    // But v_perm selector byte uses bits [2:0] for index 0-7
    // So selector_byte = 4 * S + M would give:
    // S=0, M=0-7: selector = 0-7 (bytes 0-7)
    // S=1, M=0-7: selector = 4-11 (bytes 4-7 from current pair, then... special values for 8-11?)

    // v_perm behavior for selector >= 8:
    // Looking at AMD docs... selector values 8-255 have special meanings
    // I don't think this gives us what we want.

    // Another idea: Use v_perm_b32 twice with different source pairs, controlled by computation

    // OK I think 3 v_perm might not be achievable for correct output.
    // Let me try a different hybrid approach.

    // What if we use 4 v_perm for lookups, then BITWISE operations for final combination?

    // Actually, wait. Let me re-examine the selection step.
    // We have v_even_low, v_odd_low, v_even_high, v_odd_high.
    // We need to select between low and high based on sign bit.

    // The sign bit is bit 3 of each nibble.
    // For even nibbles (nibbles 0, 2, 4, 6), this is bits 3, 11, 19, 27 of q4
    // For odd nibbles (nibbles 1, 3, 5, 7), this is bits 7, 15, 23, 31 of q4

    // Current selection uses v_perm with computed mask.
    // Alternative: Use v_cndmask_b32 with VCC computed from sign bits

    // v_cndmask_b32 dst, src0, src1, vcc
    // dst[lane] = vcc[lane] ? src1[lane] : src0[lane]

    // But this selects entire 32-bit values, not per-byte!
    // And VCC is shared across lanes.

    // Per-byte selection options:
    // 1. v_perm (1 instruction per 32-bit word)
    // 2. Mask + OR: (low & ~sign_mask) | (high & sign_mask) = 3 instructions per word
    // 3. BFI: v_bfi_b32 (1 instruction per word) - same as v_perm essentially

    // BFI is: dst = (src0 & src1) | (~src0 & src2)
    // If src0 = sign_mask, src1 = high, src2 = low:
    // dst = (sign_mask & high) | (~sign_mask & low) -- selects high where sign=1, low where sign=0

    // But we need sign_mask with 0xFF for negative bytes, 0x00 for positive bytes.
    // Computing this costs 1 v_perm per word.

    // Total with BFI: 4 v_perm (lookup) + 2 v_perm (sign expand) + 2 BFI = 6 v_perm + 2 BFI
    // That's 8 ops, same as our previous "fast" approach.

    // To get to 3 v_perm, we'd need:
    // - 2 v_perm for magnitude lookup (can't reduce further with 8-entry table)
    // - 1 v_perm for... everything else?

    // What could 1 v_perm do?
    // - Sign expansion for one word OR
    // - Selection for one word
    //
    // We need sign expansion for 2 words AND selection for 2 words.
    // 1 v_perm can't cover all of that.

    // Unless we process only one word (4 nibbles) at a time?
    // Then we'd have: 1 v_perm (mag) + 0.5 v_perm (sign expand) + 0.5 v_perm (select)?
    // That doesn't make sense since v_perm is atomic.

    // What if we use a different algorithm for one of even/odd?
    // - Even: 3 v_perm approach (optimized)
    // - Odd: Share some computation with even

    // The indices for even and odd come from different bits of q4.
    // They share the same table, but that's already in registers.
    // I don't see how to share the lookup results.

    // Let me try combining magnitude lookup with sign in a single step...

    // Unified table: 16 entries = 16 bytes = 4 registers
    const uint32_t *values = (const uint32_t *)kvalues_mxfp4;

    // The original uses q & 0x07070707 as selector (magnitude only)
    // What if we use q & 0x0F0F0F0F (full 4-bit index)?

    // This would require the selector byte to somehow select from 16 entries
    // v_perm selector byte [2:0] = index into 8 bytes
    // We'd need [3:0] = index into 16 bytes, but that's not how v_perm works.

    // What if we split differently?
    // Use 2 v_perm that together cover all 16 entries without separate selection

    // Observation: The low table (indices 0-7) and high table (indices 8-15) differ only by sign.
    // If we could apply sign in the index computation...

    // idx' = idx XOR (sign * 8) = idx ^ ((idx >> 3) << 3) = idx ^ (idx & 8) -- this doesn't change anything!

    // Actually, for indices 0-7: result is positive
    // For indices 8-15: result is negative = -(value at index - 8)

    // So: value[8+m] = -value[m] for m = 0..7

    // If we look up value[m] first, then negate based on original sign bit...
    // We're back to magnitude lookup + sign application.

    // I think 3 v_perm is not achievable with the current algorithm.
    // The fundamental issue is that we need:
    // - 2 v_perm for 4 even nibbles (8-entry table = 1 v_perm covers 4 lookups)
    // - 2 v_perm for 4 odd nibbles
    // - Something for sign handling

    // Unless... we can fold sign handling into the lookup by precomputing signed tables?

    // But that's what the original 6 v_perm does: 2 tables (pos, neg) × 2 lookups (even, odd) = 4
    // Plus 2 for selection.

    // The bottleneck is the selection. If we could eliminate selection...

    // What if we used two completely separate tables, indexed by full 4-bit index?
    // Even table: all 16 values at indices matching even nibbles
    // Odd table: all 16 values at indices matching odd nibbles
    //
    // Then: 2 v_perm for even (covers 8 indices), 2 v_perm for odd, no selection needed?
    // No, that's still 4 v_perm and doesn't eliminate the need for selection.

    // Hmm, what if we restructure to do selection DURING the lookup?

    // v_perm can use selector values 0-7 to pick from 8 bytes.
    // If we set up the source registers such that positions 0-7 contain the correct
    // values for both positive and negative (based on index modulo 8)...

    // For example:
    // src0 = {val[0], val[1], val[2], val[3]} for indices 0-3 (positive)
    // src1 = {val[0], val[9], val[10], val[11]} for negative indices 8-11
    // Wait, this doesn't make sense because the selector chooses the same byte from src0 or src1.

    // Let me think again about what problem we're really solving.

    // We have 8 nibbles (indices), each 4 bits. We want 8 bytes (values) from a 16-entry table.
    // v_perm does 4 lookups from an 8-entry table per instruction.

    // Minimum v_perm for 8 lookups from 16-entry table:
    // - If table were 8 entries: 2 v_perm (one for even nibbles, one for odd)
    // - Since table is 16 entries, each lookup needs 2 v_perm + selection: 2*(2+1) = 6 v_perm
    //
    // The 6 v_perm lower bound comes from: 4 lookups × (2 + 1 select) = 6, but we share lookups:
    // - 4 v_perm for lookups (each covers 4 nibbles × 2 table halves = 8 lookups... wait, no)

    // Let me count more carefully:
    // - v_perm #1: even nibbles, low table (4 lookups)
    // - v_perm #2: even nibbles, high table (4 lookups)
    // - v_perm #3: odd nibbles, low table (4 lookups)
    // - v_perm #4: odd nibbles, high table (4 lookups)
    // - v_perm #5: select for even (combines #1 and #2)
    // - v_perm #6: select for odd (combines #3 and #4)
    //
    // Total lookups: 16 (from 4 v_perm), but we only need 8 final values.
    // The selection discards half the looked-up values.

    // If we could somehow only look up the values we need...
    // But the 8-entry limit of v_perm prevents this for a 16-entry table.

    // What if we changed the table to 8 entries + sign bit?
    // That's the magnitude approach: 2 v_perm for magnitude, then sign application.
    // Sign application costs at least 2 v_perm (for expansion) + 2 BFI (for selection) = 4 ops.
    // Total: 2 + 4 = 6 ops (same as original if we count BFI as free).
    // But BFI is not v_perm, so maybe this is "3 v_perm + 3 other ops"?

    // Wait, the user said "3 permute instructions" with "DPP or other stuff".
    // Maybe they mean 3 v_perm total, with sign handling done by other means.

    // 3 v_perm breakdown:
    // - 2 v_perm for magnitude lookup (even + odd, 8-entry table)
    // - 1 v_perm for... sign expansion of one word? Or some combined operation?

    // What if we use DPP for sign expansion?
    // DPP does data movement between lanes. Sign expansion needs data transformation within a lane.
    // These are different operations.

    // Unless we distribute work across lanes... but that complicates the algorithm significantly.

    // Let me try: 2 v_perm (magnitude) + 1 v_perm (sign for one word) + ALU for the rest

    const uint32_t mag_lo = 0x03020100;
    const uint32_t mag_hi = 0x0c080604;

    uint32_t q_even = q4;
    uint32_t q_odd = q4 >> 4;

    // 2 v_perm for magnitude
    uint32_t mag_even = __builtin_amdgcn_perm(mag_hi, mag_lo, q_even & 0x07070707);
    uint32_t mag_odd = __builtin_amdgcn_perm(mag_hi, mag_lo, q_odd & 0x07070707);

    // Sign bits
    uint32_t sign_even = (q_even >> 3) & 0x01010101;
    uint32_t sign_odd = (q_odd >> 3) & 0x01010101;

    // 1 v_perm for sign expansion (even only)
    const uint32_t expand = 0x0000FF00;
    uint32_t mask_even = __builtin_amdgcn_perm(expand, expand, sign_even);

    // For odd, use arithmetic: mask = 0 - sign (with carry issues)
    // Actually this doesn't work due to carry.

    // Alternative: replicate the v_perm pattern with shifts
    // If sign_odd = 0x01010101, we want mask_odd = 0xFFFFFFFF
    // If sign_odd = 0x00000000, we want mask_odd = 0x00000000
    // If sign_odd = 0x01000001, we want mask_odd = 0xFF0000FF

    // Using v_perm costs 1 op. Can we match with ALU?
    // mask = sign * 0xFF per byte... need per-byte multiply
    // Or: mask = (sign << 8) - sign per 16-bit half... multiple ops

    // Simplest: just use v_perm for odd too (total 4 v_perm for signs, not 3)
    uint32_t mask_odd = __builtin_amdgcn_perm(expand, expand, sign_odd);

    // Apply sign
    uint32_t neg_even = (~mag_even + 0x01010101);  // Carry issues!
    uint32_t neg_odd = (~mag_odd + 0x01010101);

    uint32_t res_x, res_y;
    asm volatile("v_bfi_b32 %0, %1, %2, %3" : "=v"(res_x) : "v"(mask_even), "v"(neg_even), "v"(mag_even));
    asm volatile("v_bfi_b32 %0, %1, %2, %3" : "=v"(res_y) : "v"(mask_odd), "v"(neg_odd), "v"(mag_odd));

    return make_int2(res_x, res_y);
    // This has carry issues from the negation. Not correct!
}

// ============================================================================
// Test kernel
// ============================================================================
__global__ void test_approaches(uint32_t* inputs, int2* ref_results,
                                 int2* a1_results, int2* a4_results,
                                 int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    uint32_t q4 = inputs[idx];
    ref_results[idx] = lookup_reference(q4);
    a1_results[idx] = lookup_approach1(q4);  // Will have carry issues
    a4_results[idx] = lookup_approach4(q4);  // pk_sub approach
}

// ============================================================================
// Host code
// ============================================================================
void check_hip(hipError_t err, const char* msg) {
    if (err != hipSuccess) {
        fprintf(stderr, "HIP Error: %s: %s\n", msg, hipGetErrorString(err));
        exit(1);
    }
}

int main() {
    printf("=== MXFP4 3 v_perm Exploration ===\n\n");

    const int N = 65536;
    const int BLOCK_SIZE = 256;
    const int GRID_SIZE = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Allocate
    uint32_t* h_inputs = new uint32_t[N];
    int2* h_ref = new int2[N];
    int2* h_a1 = new int2[N];
    int2* h_a4 = new int2[N];

    for (int i = 0; i < N; i++) {
        h_inputs[i] = i;
    }

    uint32_t* d_inputs;
    int2* d_ref, *d_a1, *d_a4;

    check_hip(hipMalloc(&d_inputs, N * sizeof(uint32_t)), "malloc");
    check_hip(hipMalloc(&d_ref, N * sizeof(int2)), "malloc");
    check_hip(hipMalloc(&d_a1, N * sizeof(int2)), "malloc");
    check_hip(hipMalloc(&d_a4, N * sizeof(int2)), "malloc");

    check_hip(hipMemcpy(d_inputs, h_inputs, N * sizeof(uint32_t), hipMemcpyHostToDevice), "memcpy");

    // Run
    hipLaunchKernelGGL(test_approaches, dim3(GRID_SIZE), dim3(BLOCK_SIZE), 0, 0,
                       d_inputs, d_ref, d_a1, d_a4, N);
    check_hip(hipDeviceSynchronize(), "kernel");

    // Copy back
    check_hip(hipMemcpy(h_ref, d_ref, N * sizeof(int2), hipMemcpyDeviceToHost), "memcpy");
    check_hip(hipMemcpy(h_a1, d_a1, N * sizeof(int2), hipMemcpyDeviceToHost), "memcpy");
    check_hip(hipMemcpy(h_a4, d_a4, N * sizeof(int2), hipMemcpyDeviceToHost), "memcpy");

    // Check approach 1
    printf("[Approach 1] Magnitude + BFI (carry issues expected):\n");
    int errors_a1 = 0;
    for (int i = 0; i < N && errors_a1 < 5; i++) {
        if (h_ref[i].x != h_a1[i].x || h_ref[i].y != h_a1[i].y) {
            if (errors_a1 == 0) printf("  First mismatches:\n");
            printf("    i=%d (0x%04x): ref=(0x%08x,0x%08x) got=(0x%08x,0x%08x)\n",
                   i, h_inputs[i], h_ref[i].x, h_ref[i].y, h_a1[i].x, h_a1[i].y);
            errors_a1++;
        }
    }
    if (errors_a1 == 0) printf("  PASSED!\n");
    else printf("  FAILED: %d+ errors\n", errors_a1);

    // Check approach 4 (pk_sub)
    printf("\n[Approach 4] pk_sub sign expansion:\n");
    int errors_a4 = 0;
    for (int i = 0; i < N && errors_a4 < 5; i++) {
        if (h_ref[i].x != h_a4[i].x || h_ref[i].y != h_a4[i].y) {
            if (errors_a4 == 0) printf("  First mismatches:\n");
            printf("    i=%d (0x%04x): ref=(0x%08x,0x%08x) got=(0x%08x,0x%08x)\n",
                   i, h_inputs[i], h_ref[i].x, h_ref[i].y, h_a4[i].x, h_a4[i].y);
            errors_a4++;
        }
    }
    if (errors_a4 == 0) printf("  PASSED!\n");
    else printf("  FAILED: %d+ errors\n", errors_a4);

    // Cleanup
    hipFree(d_inputs);
    hipFree(d_ref);
    hipFree(d_a1);
    hipFree(d_a4);
    delete[] h_inputs;
    delete[] h_ref;
    delete[] h_a1;
    delete[] h_a4;

    printf("\n=== Done ===\n");
    return 0;
}
