// Test: Can we compute MXFP4 values instead of table lookup?
// Compile: g++ -o test_mxfp4_compute test_mxfp4_compute.cpp
// Run: ./test_mxfp4_compute

#include <cstdio>
#include <cstdint>

// Original table
static const int8_t kvalues_mxfp4[16] = {
    0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12
};

// Compute MXFP4 value from 4-bit index
// e2m1 format: bit3=sign, bits0-2=magnitude encoding
inline int8_t compute_mxfp4(int idx) {
    // Extract sign and magnitude index
    int sign = (idx >> 3) & 1;      // bit 3
    int mag_idx = idx & 0x7;        // bits 0-2

    // Magnitude lookup (only 8 values)
    // 0->0, 1->1, 2->2, 3->3, 4->4, 5->6, 6->8, 7->12
    int8_t mag;
    if (mag_idx < 4) {
        mag = mag_idx;  // 0,1,2,3
    } else {
        // 4->4, 5->6, 6->8, 7->12
        // Pattern: 2^(mag_idx-2) for mag_idx=4,6 and 1.5*2^(mag_idx-3) for mag_idx=5,7
        // Simpler: use small lookup or bit tricks
        static const int8_t high_mags[4] = {4, 6, 8, 12};
        mag = high_mags[mag_idx - 4];
    }

    return sign ? -mag : mag;
}

// Alternative: Pure bit manipulation (no lookup at all)
inline int8_t compute_mxfp4_v2(int idx) {
    int sign = (idx >> 3) & 1;
    int m = idx & 0x7;

    // For m < 4: result = m
    // For m >= 4: result = 4 + 2*(m-4) + ((m-4)>>1)*2
    //           = 4 + 2*(m-4) + (m>=6 ? 2 : 0)
    // Actually: 4->4, 5->6, 6->8, 7->12
    // Pattern: 4->4 (+0), 5->6 (+2), 6->8 (+4), 7->12 (+8)
    // Increment: 0, 2, 4, 8 = 2^max(0, m-5)

    int8_t mag;
    if (m < 4) {
        mag = m;
    } else {
        // 4 + (m-4)*2 + extra
        // m=4: 4 + 0 = 4
        // m=5: 4 + 2 = 6
        // m=6: 4 + 4 = 8
        // m=7: 4 + 4 + 4 = 12 (extra 4 for m=7)
        mag = 4 + (m - 4) * 2 + ((m == 7) ? 4 : 0);
    }

    return sign ? -mag : mag;
}

// GPU-friendly version using only bit ops (no branches)
inline int8_t compute_mxfp4_v3(int idx) {
    int sign = (idx >> 3) & 1;
    int m = idx & 0x7;

    // Branchless magnitude computation
    // mag_low = m (for m < 4)
    // mag_high = 4 + 2*(m-4) + 4*(m==7) = 2*m - 4 + 4*(m==7)
    // select based on m >= 4

    int is_high = (m >> 2) & 1;  // 1 if m >= 4
    int is_7 = (m == 7) ? 1 : 0; // Need this for the +4 at m=7

    int mag_low = m;
    int mag_high = 2*m - 4 + 4*is_7;

    int8_t mag = is_high ? mag_high : mag_low;

    return sign ? -mag : mag;
}

// Fully branchless version
// Pattern for high values (m >= 4):
// m=4: 4, m=5: 6, m=6: 8, m=7: 12
// Differences: 4->5: +2, 5->6: +2, 6->7: +4
// Alternative view: base=4, add 2*(m-4) for m=4,5,6, add 2*(m-4)+4 for m=7
// Or: 4 + 2*(m-4) + 2*((m&3)==3)
inline int8_t compute_mxfp4_branchless(int idx) {
    int sign = (idx >> 3) & 1;
    int m = idx & 0x7;

    int is_high = (m >> 2);           // 1 if m >= 4, 0 otherwise
    int m_low = m & 3;                // m mod 4
    int is_3 = (m_low == 3) ? 1 : 0;  // Extra +2 for m=3 in high range (i.e., m=7)

    // For low (m < 4): mag = m
    // For high (m >= 4): mag = 4 + 2*(m-4) + 2*is_3 = 4 + 2*m_low + 2*is_3
    int mag_high = 4 + 2*m_low + 2*is_3;
    int mag = m * (1 - is_high) + mag_high * is_high;

    // Apply sign: val = sign ? -mag : mag = mag * (1 - 2*sign)
    return (int8_t)(mag * (1 - 2*sign));
}

// Pure bit manipulation version - no comparisons at all
// Key insight: for m=0..7, values are 0,1,2,3,4,6,8,12
// m_low = m & 3 gives 0,1,2,3,0,1,2,3
// For m >= 4: mag = 4 + 2*m_low + 2*(m_low==3)
// To detect m_low==3 without comparison: (m_low & 1) & (m_low >> 1) = 1 only when m_low=3
inline int8_t compute_mxfp4_bitops(int idx) {
    int sign = (idx >> 3) & 1;
    int m = idx & 0x7;

    int is_high = (m >> 2) & 1;       // 1 if m >= 4
    int m_low = m & 3;                // m mod 4

    // Detect m_low == 3: both bits set
    int is_3 = (m_low & 1) & (m_low >> 1);  // 1 only when m_low=3 (binary 11)

    // For m < 4: mag = m
    // For m >= 4: mag = 4 + 2*m_low + 2*is_3
    int mag_high = 4 + 2*m_low + 2*is_3;
    int mag = m * (1 - is_high) + mag_high * is_high;

    // Apply sign
    return (int8_t)(mag * (1 - 2*sign));
}

int main() {
    printf("Testing MXFP4 computation vs table lookup:\n\n");
    printf("idx | table | v1 | branchless | bitops | match?\n");
    printf("----|-------|----|-----------:|-------:|-------\n");

    bool all_match = true;
    for (int i = 0; i < 16; i++) {
        int8_t table_val = kvalues_mxfp4[i];
        int8_t v1 = compute_mxfp4(i);
        int8_t vb = compute_mxfp4_branchless(i);
        int8_t vbit = compute_mxfp4_bitops(i);

        bool match = (v1 == table_val) && (vb == table_val) && (vbit == table_val);
        all_match &= match;

        printf("%3d | %5d | %2d | %10d | %6d | %s\n",
               i, table_val, v1, vb, vbit, match ? "OK" : "FAIL");
    }

    printf("\nAll match: %s\n", all_match ? "YES" : "NO");

    if (all_match) {
        printf("\n=== GPU-friendly formula (pure bit ops) ===\n");
        printf("sign = (idx >> 3) & 1\n");
        printf("m = idx & 0x7\n");
        printf("is_high = (m >> 2) & 1\n");
        printf("m_low = m & 3\n");
        printf("is_3 = (m_low & 1) & (m_low >> 1)  // detect m_low == 3\n");
        printf("mag_high = 4 + 2*m_low + 2*is_3\n");
        printf("mag = m * (1 - is_high) + mag_high * is_high\n");
        printf("result = mag * (1 - 2*sign)\n");
    }

    return all_match ? 0 : 1;
}
