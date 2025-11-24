#pragma once

// Note: WARP_SIZE and other basic definitions come from parent common.cuh
// This file is included FROM common.cuh, so those definitions are already available

#include "gfx906-config.h"

// ================================================================================================
// AMD GFX906 DPP (Data Parallel Primitives) Warp Reductions
// ================================================================================================
// Hardware-specific warp reduction implementations using AMD DPP instructions
// Faster than __shfl_xor_sync on GFX906 architecture
//
// This implementation achieves maximum performance by using fused DPP+operation instructions
// (e.g., v_add_f32_dpp, v_max_f32_dpp) instead of separate shuffle and arithmetic operations.
// The number of cycles needed for every shfl operation has been minimized.
// (Optimal for gfx906 only).
//
// Key optimizations:
// - XOR 1, 2, 8: Fused DPP+operation (1 instruction each) via inline assembly
// - XOR 4: Dual-bank shuffle (can't fuse, needs 2 DPP + 1 ALU)
// - XOR 16: DS_SWIZZLE (can't fuse, uses LDS)
// - Operation tags (AddOp, MaxOp) for compile-time dispatch with zero overhead
// - Exact barrier scheme from old inline assembly preserved
// ================================================================================================

#ifdef GGML_USE_HIP

// ============================================================================
// Macro to generate fused DPP+operation instruction wrappers for float
// ============================================================================
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

// XOR 1 - FIRST operation in reduction (s_nop 4 for EXEC hazard protection)
DEFINE_FUSED_DPP_F32(hip_add_xor1_f32, "s_nop 4\n", "quad_perm:[1,0,3,2]", "v_add_f32_dpp")
DEFINE_FUSED_DPP_F32(hip_max_xor1_f32, "s_nop 4\n", "quad_perm:[1,0,3,2]", "v_max_f32_dpp")

// XOR 2 - Subsequent operation (s_nop 1 for VGPR→DPP hazard)
DEFINE_FUSED_DPP_F32(hip_add_xor2_f32, "s_nop 1\n", "quad_perm:[2,3,0,1]", "v_add_f32_dpp")
DEFINE_FUSED_DPP_F32(hip_max_xor2_f32, "s_nop 1\n", "quad_perm:[2,3,0,1]", "v_max_f32_dpp")

// XOR 8 - Subsequent operation (s_nop 1)
DEFINE_FUSED_DPP_F32(hip_add_xor8_f32, "s_nop 1\n", "row_ror:8", "v_add_f32_dpp")
DEFINE_FUSED_DPP_F32(hip_max_xor8_f32, "s_nop 1\n", "row_ror:8", "v_max_f32_dpp")

#undef DEFINE_FUSED_DPP_F32

// XOR 4 - Dual bank shuffle (can't fuse, needs two DPP moves)
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

// XOR 16 - DS_SWIZZLE (can't fuse, uses LDS)
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

// ============================================================================
// Operation tags for compile-time dispatch
// ============================================================================
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

// ============================================================================
// Generic warp reduction using fused DPP operations (float only)
// ============================================================================
template<int width = WARP_SIZE, typename Op>
static __device__ __forceinline__ float warp_reduce_amd_f32(float x) {
    // First DPP needs s_nop 4, subsequent need s_nop 1
    if (width >= 2)  x = Op::xor1(x);  // XOR 1: FIRST DPP, fused (1 inst, s_nop 4)
    if (width >= 4)  x = Op::xor2(x);  // XOR 2: Subsequent, fused (1 inst, s_nop 1)
    if (width >= 8)  x = Op::apply(x, hip_shuffle_xor4_f32(x));  // XOR 4: Can't fuse (3+1 inst, s_nop 1)
    if (width >= 16) x = Op::xor8(x);  // XOR 8: Subsequent, fused (1 inst, s_nop 1)
    if (width >= 32) x = Op::apply(x, hip_shuffle_xor16_f32(x)); // XOR 16: ds_swizzle (2+1 inst, s_waitcnt)
    if (width == 64) x = Op::apply(x, __shfl_xor(x, 32, 64));    // XOR 32: cross-wave for wavefront-64
    return x;
}

// ============================================================================
// Fallback shuffle functions for non-float types (int, half2, etc.)
// ============================================================================
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

// ============================================================================
// Unified shuffle XOR operation - works on both NVIDIA and AMD
// For use by other GFX906 optimizations
// ============================================================================
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

// ============================================================================
// GFX906-optimized warp reduction functions
// These are used by common.cuh to provide hardware-optimized implementations
// ============================================================================

// Sum reduction for float (using fused DPP)
template<int width = WARP_SIZE>
static __device__ __forceinline__ float gfx906_warp_reduce_sum_f32(float x) {
    return warp_reduce_amd_f32<width, AddOp>(x);
}

// Max reduction for float (using fused DPP)
template<int width = WARP_SIZE>
static __device__ __forceinline__ float gfx906_warp_reduce_max_f32(float x) {
    return warp_reduce_amd_f32<width, MaxOp>(x);
}

// Generic reduction for int and other types (using DPP shuffle)
template<int width = WARP_SIZE, typename T>
static __device__ __forceinline__ T gfx906_warp_reduce_sum_generic(T x) {
    #pragma unroll
    for (int offset = width/2; offset > 0; offset >>= 1) {
        x += gfx906_shfl_xor_sync<width>(x, offset);
    }
    return x;
}

#endif // GGML_USE_HIP
