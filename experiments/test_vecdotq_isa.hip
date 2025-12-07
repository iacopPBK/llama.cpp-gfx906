// Test to verify vecdotq.cuh ISA output
#include <hip/hip_runtime.h>
#include <cstdint>

// Copy of current vecdotq.cuh implementation
static __device__ __forceinline__ int2 get_int_from_table_16_current(const int & q4, const int8_t * table) {
    const uint32_t *values = (const uint32_t *)table;

    const uint32_t q_even = q4;
    const uint32_t q_odd  = (q4 >> 4);

    const uint32_t sel_even = q_even & 0x07070707;
    const uint32_t sel_odd  = q_odd & 0x07070707;

    uint32_t v_even_low = __builtin_amdgcn_perm(values[1], values[0], sel_even);
    uint32_t v_odd_low = __builtin_amdgcn_perm(values[1], values[0], sel_odd);
    uint32_t v_even_high = __builtin_amdgcn_perm(values[3], values[2], sel_even);
    uint32_t v_odd_high = __builtin_amdgcn_perm(values[3], values[2], sel_odd);

    uint32_t b3e = (q_even >> 3) & 0x01010101;
    uint32_t me = b3e; me |= me << 1; me |= me << 2; me |= me << 4;

    uint32_t b3o = (q_odd >> 3) & 0x01010101;
    uint32_t mo = b3o; mo |= mo << 1; mo |= mo << 2; mo |= mo << 4;

    uint32_t res_x = (v_even_high & me) | (v_even_low & ~me);
    uint32_t res_y = (v_odd_high & mo) | (v_odd_low & ~mo);

    return make_int2(res_x, res_y);
}

// v3 from test_8entry_lookup
static __device__ __forceinline__ int2 lookup_8entry_v3(const int q4, const uint32_t* half_values, const uint32_t* neg_half_values) {
    const uint32_t q_even = q4;
    const uint32_t q_odd  = (q4 >> 4);

    const uint32_t idx_even = q_even & 0x07070707;
    const uint32_t idx_odd  = q_odd & 0x07070707;
    const uint32_t sign_even = (q_even >> 3) & 0x01010101;
    const uint32_t sign_odd  = (q_odd >> 3) & 0x01010101;

    uint32_t pos_even = __builtin_amdgcn_perm(half_values[1], half_values[0], idx_even);
    uint32_t pos_odd  = __builtin_amdgcn_perm(half_values[1], half_values[0], idx_odd);
    uint32_t neg_even = __builtin_amdgcn_perm(neg_half_values[1], neg_half_values[0], idx_even);
    uint32_t neg_odd  = __builtin_amdgcn_perm(neg_half_values[1], neg_half_values[0], idx_odd);

    uint32_t sm_even = sign_even; sm_even |= sm_even<<1; sm_even |= sm_even<<2; sm_even |= sm_even<<4;
    uint32_t sm_odd  = sign_odd;  sm_odd  |= sm_odd<<1;  sm_odd  |= sm_odd<<2;  sm_odd  |= sm_odd<<4;

    uint32_t res_even = (neg_even & sm_even) | (pos_even & ~sm_even);
    uint32_t res_odd  = (neg_odd  & sm_odd)  | (pos_odd  & ~sm_odd);

    return make_int2(res_even, res_odd);
}

__global__ void test_current(const int* q4_data, int2* output, const int8_t* table, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    output[idx] = get_int_from_table_16_current(q4_data[idx], table);
}

__global__ void test_v3(const int* q4_data, int2* output, const int8_t* half, const int8_t* neg_half, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    output[idx] = lookup_8entry_v3(q4_data[idx], (const uint32_t*)half, (const uint32_t*)neg_half);
}

int main() { return 0; }
