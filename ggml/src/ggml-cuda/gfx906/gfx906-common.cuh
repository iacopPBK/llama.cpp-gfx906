#pragma once

// DPP-based warp reductions for GFX906
// Fuses shuffle + ALU into single DPP instruction, reducing latency

#include "gfx906-config.h"

#ifdef GGML_USE_HIP

#define DEFINE_FUSED_DPP_F32(name, barrier, dpp_ctrl, vop_instr)           \
    static __device__ __forceinline__ float name(float x) {                \
        float result;                                                       \
        asm volatile(                                                       \
            barrier                                                         \
            vop_instr " %0, %1, %1 " dpp_ctrl " row_mask:0xf bank_mask:0xf" \
            : "=v"(result) : "v"(x) : "memory"                             \
        );                                                                  \
        return result;                                                      \
    }

DEFINE_FUSED_DPP_F32(hip_add_xor1_f32, "s_nop 4\n", "quad_perm:[1,0,3,2]", "v_add_f32_dpp")
DEFINE_FUSED_DPP_F32(hip_max_xor1_f32, "s_nop 4\n", "quad_perm:[1,0,3,2]", "v_max_f32_dpp")

DEFINE_FUSED_DPP_F32(hip_add_xor2_f32, "s_nop 1\n", "quad_perm:[2,3,0,1]", "v_add_f32_dpp")
DEFINE_FUSED_DPP_F32(hip_max_xor2_f32, "s_nop 1\n", "quad_perm:[2,3,0,1]", "v_max_f32_dpp")

DEFINE_FUSED_DPP_F32(hip_add_xor8_f32, "s_nop 1\n", "row_ror:8", "v_add_f32_dpp")
DEFINE_FUSED_DPP_F32(hip_max_xor8_f32, "s_nop 1\n", "row_ror:8", "v_max_f32_dpp")

#undef DEFINE_FUSED_DPP_F32

static __device__ __forceinline__ float hip_shuffle_xor4_f32(float x) {
    int v_src = __float_as_int(x);
    int v_dst;
    asm volatile(
        "v_mov_b32 %0, %1\n"
        "s_nop 1\n"
        "v_mov_b32_dpp %0, %1 row_shl:4 row_mask:0xf bank_mask:0x5\n"
        "v_mov_b32_dpp %0, %1 row_shr:4 row_mask:0xf bank_mask:0xa\n"
        : "=v"(v_dst) : "v"(v_src) : "memory"
    );
    return __int_as_float(v_dst);
}

static __device__ __forceinline__ float hip_shuffle_xor16_f32(float x) {
    int int_val = __float_as_int(x);
    int result;
    asm volatile(
        "ds_swizzle_b32 %0, %1 offset:swizzle(SWAP,16)\n"
        "s_waitcnt lgkmcnt(0)\n"
        : "=v"(result) : "v"(int_val) : "memory"
    );
    return __int_as_float(result);
}

struct AddOp {
    static __device__ __forceinline__ float apply(float a, float b) { return a + b; }
    static __device__ __forceinline__ float xor1(float x) { return hip_add_xor1_f32(x); }
    static __device__ __forceinline__ float xor2(float x) { return hip_add_xor2_f32(x); }
    static __device__ __forceinline__ float xor8(float x) { return hip_add_xor8_f32(x); }
};

struct MaxOp {
    static __device__ __forceinline__ float apply(float a, float b) { return fmaxf(a, b); }
    static __device__ __forceinline__ float xor1(float x) { return hip_max_xor1_f32(x); }
    static __device__ __forceinline__ float xor2(float x) { return hip_max_xor2_f32(x); }
    static __device__ __forceinline__ float xor8(float x) { return hip_max_xor8_f32(x); }
};

template<int width = WARP_SIZE, typename Op>
static __device__ __forceinline__ float warp_reduce_amd_f32(float x) {
    if (width >= 2)  x = Op::xor1(x);
    if (width >= 4)  x = Op::xor2(x);
    if (width >= 8)  x = Op::apply(x, hip_shuffle_xor4_f32(x));
    if (width >= 16) x = Op::xor8(x);
    if (width >= 32) x = Op::apply(x, hip_shuffle_xor16_f32(x));
    if (width == 64) x = Op::apply(x, __shfl_xor(x, 32, 64));
    return x;
}

template<typename T>
static __device__ __forceinline__ T hip_dpp_xor1(T value) {
    static_assert(sizeof(T) == 4, "DPP operations require 32-bit types");
    int int_val = *reinterpret_cast<int*>(&value);
    int result;
    asm volatile(
        "s_nop 4\n"
        "v_mov_b32_dpp %0, %1 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf"
        : "=v"(result) : "v"(int_val) : "memory"
    );
    return *reinterpret_cast<T*>(&result);
}

template<typename T>
static __device__ __forceinline__ T hip_dpp_xor2(T value) {
    static_assert(sizeof(T) == 4, "DPP operations require 32-bit types");
    int int_val = *reinterpret_cast<int*>(&value);
    int result;
    asm volatile(
        "s_nop 1\n"
        "v_mov_b32_dpp %0, %1 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf"
        : "=v"(result) : "v"(int_val) : "memory"
    );
    return *reinterpret_cast<T*>(&result);
}

template<typename T>
static __device__ __forceinline__ T hip_dpp_xor4(T value) {
    static_assert(sizeof(T) == 4, "DPP operations require 32-bit types");
    int v_src = *reinterpret_cast<int*>(&value);
    int v_dst;
    asm volatile(
        "v_mov_b32 %0, %1\n"
        "s_nop 1\n"
        "v_mov_b32_dpp %0, %1 row_shl:4 row_mask:0xf bank_mask:0x5\n"
        "v_mov_b32_dpp %0, %1 row_shr:4 row_mask:0xf bank_mask:0xa\n"
        : "=v"(v_dst) : "v"(v_src) : "memory"
    );
    return *reinterpret_cast<T*>(&v_dst);
}

template<typename T>
static __device__ __forceinline__ T hip_dpp_xor8(T value) {
    static_assert(sizeof(T) == 4, "DPP operations require 32-bit types");
    int int_val = *reinterpret_cast<int*>(&value);
    int result;
    asm volatile(
        "s_nop 1\n"
        "v_mov_b32_dpp %0, %1 row_ror:8 row_mask:0xf bank_mask:0xf"
        : "=v"(result) : "v"(int_val) : "memory"
    );
    return *reinterpret_cast<T*>(&result);
}

template<typename T>
static __device__ __forceinline__ T hip_dpp_xor16(T value) {
    static_assert(sizeof(T) == 4, "DPP operations require 32-bit types");
    int int_val = *reinterpret_cast<int*>(&value);
    int result;
    asm volatile(
        "ds_swizzle_b32 %0, %1 offset:swizzle(SWAP,16)\n"
        "s_waitcnt lgkmcnt(0)\n"
        : "=v"(result) : "v"(int_val) : "memory"
    );
    return *reinterpret_cast<T*>(&result);
}

template<int width = WARP_SIZE, typename T>
static __device__ __forceinline__ T gfx906_shfl_xor_sync(T x, int offset) {
    switch (~offset) {
        case ~1:  return hip_dpp_xor1(x);
        case ~2:  return hip_dpp_xor2(x);
        case ~4:  return hip_dpp_xor4(x);
        case ~8:  return hip_dpp_xor8(x);
        case ~16: return hip_dpp_xor16(x);
        default:  return __shfl_xor(x, offset, width);
    }
}

template<int width = WARP_SIZE>
static __device__ __forceinline__ float gfx906_warp_reduce_sum_f32(float x) {
    return warp_reduce_amd_f32<width, AddOp>(x);
}

template<int width = WARP_SIZE>
static __device__ __forceinline__ float gfx906_warp_reduce_max_f32(float x) {
    return warp_reduce_amd_f32<width, MaxOp>(x);
}

template<int width = WARP_SIZE, typename T>
static __device__ __forceinline__ T gfx906_warp_reduce_sum_generic(T x) {
    #pragma unroll
    for (int offset = width/2; offset > 0; offset >>= 1) {
        x += gfx906_shfl_xor_sync<width>(x, offset);
    }
    return x;
}

#endif // GGML_USE_HIP
