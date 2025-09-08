#pragma once

// GFX906 Memory Access ISA-Level Implementation
// Low-level GCN assembly implementations for vectorized loads, coalesced access,
// and LDS double buffering using Vega 20nm ISA primitives

#ifdef GGML_HIP_GFX906_OPTIMIZED

#    include "gfx906-config.cuh"

#    include <hip/hip_runtime.h>

namespace gfx906 {
namespace memory_isa {

// ================================
// Vectorized 128-bit Loads (DWORDX4)
// ================================

// BUFFER_LOAD_DWORDX4 - Load 4 consecutive dwords (128 bits)
__device__ __forceinline__ float4 buffer_load_dwordx4(const float * __restrict__ addr, uint32_t offset = 0) {
    float4 result;

    // Inline assembly for BUFFER_LOAD_DWORDX4
    // v[dst:dst+3] = buffer_load_dwordx4 v[addr], s[desc:desc+3], offset
    asm volatile(
        "buffer_load_dwordx4 %0, %1, s[0:3], %2 offen\n\t"
        "s_waitcnt vmcnt(0)"
        : "=v"(result)
        : "v"(addr), "i"(offset)
        : "memory");

    return result;
}

// GLOBAL_LOAD_DWORDX4 - Global memory 128-bit load (with offset)
__device__ __forceinline__ float4 global_load_dwordx4(const float * __restrict__ addr) {
    float4 result;

    // Ensure 16-byte alignment for optimal performance
    assert(((uintptr_t) addr & 0xF) == 0);

    // For direct pointer access, we can still use the regular load
    // The compiler will generate the proper instruction
    result = *((const float4*)addr);
    __builtin_amdgcn_s_waitcnt(0x3F70); // vmcnt(0)

    return result;
}

// GLOBAL_LOAD_DWORDX4 with explicit base+offset addressing using inline assembly
__device__ __forceinline__ float4 global_load_dwordx4_offset(const float * __restrict__ base, uint32_t offset) {
    float4 result;
    
    // Correct inline assembly using SGPR base + VGPR offset
    asm volatile(
        "global_load_dwordx4 %0, %1, %2\n\t"
        "s_waitcnt vmcnt(0)"
        : "=v"(result)           // Output: data in VGPR
        : "v"(offset),           // Input: 32-bit offset in VGPR
          "s"(base)              // Input: base pointer in SGPR pair
        : "memory");
    
    return result;
}

// FLAT_LOAD_DWORDX4 - Flat address space 128-bit load
__device__ __forceinline__ float4 flat_load_dwordx4(const float * __restrict__ addr) {
    float4 result;

    asm volatile(
        "flat_load_dwordx4 %0, %1\n\t"
        "s_waitcnt vmcnt(0)"
        : "=v"(result)
        : "v"(addr)
        : "memory");

    return result;
}

// Load 8 half-precision values (128 bits) as packed data
__device__ __forceinline__ uint4 global_load_half8(const __half * __restrict__ addr) {
    uint4 result;

    asm volatile(
        "global_load_dwordx4 %0, %1, off\n\t"
        "s_waitcnt vmcnt(0)"
        : "=v"(result)
        : "v"(addr)
        : "memory");

    return result;
}

// ================================
// Coalesced Memory Access Patterns
// ================================

// Calculate coalesced address for current thread
__device__ __forceinline__ uintptr_t calculate_coalesced_address(const void * base_addr, size_t element_size) {
    // Get thread ID within wavefront (0-63)
    const uint32_t lane_id = __builtin_amdgcn_mbcnt_hi(~0u, __builtin_amdgcn_mbcnt_lo(~0u, 0u));

    // Calculate per-thread address for coalescing
    // address_for_thread_N = base_address + (N * element_size)
    return (uintptr_t) base_addr + (lane_id * element_size);
}

// Coalesced DWORD load across wavefront
__device__ __forceinline__ float coalesced_load_dword(const float * __restrict__ base_addr) {
    // Each thread loads from its coalesced position
    const float * thread_addr = base_addr + (__builtin_amdgcn_mbcnt_hi(~0u, __builtin_amdgcn_mbcnt_lo(~0u, 0u)));

    float result;
    asm volatile(
        "global_load_dword %0, %1, off\n\t"
        "s_waitcnt vmcnt(0)"
        : "=v"(result)
        : "v"(thread_addr)
        : "memory");

    return result;
}

// Coalesced DWORDX4 load across wavefront (256 bytes total)
__device__ __forceinline__ float4 coalesced_load_dwordx4(const float * __restrict__ base_addr) {
    // Each thread loads 16 bytes, wavefront loads 1024 bytes total
    const uint32_t lane_id     = __builtin_amdgcn_mbcnt_hi(~0u, __builtin_amdgcn_mbcnt_lo(~0u, 0u));
    const float *  thread_addr = base_addr + (lane_id * 4);

    return global_load_dwordx4(thread_addr);
}

// ================================
// LDS Double Buffering with ISA
// ================================

// Direct LDS load bypassing VGPRs (asynchronous)
__device__ __forceinline__ void buffer_load_to_lds_async(const void * __restrict__ buffer_addr,
                                                         void * __restrict__ lds_addr,
                                                         uint32_t num_dwords) {
    // BUFFER_STORE_LDS_DWORD - Actually loads from buffer to LDS
    // This is asynchronous and increments lgkmcnt
    asm volatile("buffer_load_dword %0, %1, s[0:3], 0 offen lds\n\t" : : "v"(lds_addr), "v"(buffer_addr) : "memory");
}

// Wait for LDS operations to complete
__device__ __forceinline__ void wait_for_lds() {
    // S_WAITCNT lgkmcnt(0) - Wait for all LDS operations
    asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
}

// Workgroup barrier synchronization
__device__ __forceinline__ void workgroup_barrier() {
    // S_BARRIER - Synchronize all wavefronts in workgroup
    asm volatile("s_barrier" ::: "memory");
}

// LDS double buffer implementation with ISA primitives
template <typename T, int BUFFER_SIZE> class LDSDoubleBufferISA {
  private:
    // Two buffers in LDS
    __shared__ T buffer_a[BUFFER_SIZE];
    __shared__ T buffer_b[BUFFER_SIZE];
    bool         current_is_a;

  public:
    __device__ LDSDoubleBufferISA() : current_is_a(true) {}

    // Get current compute buffer
    __device__ __forceinline__ T * get_compute_buffer() { return current_is_a ? buffer_a : buffer_b; }

    // Get current prefetch buffer
    __device__ __forceinline__ T * get_prefetch_buffer() { return current_is_a ? buffer_b : buffer_a; }

    // Async prefetch to next buffer using MUBUF with LDS=1
    __device__ __forceinline__ void prefetch_async(const T * __restrict__ global_addr, int elements) {
        T *       lds_buffer = get_prefetch_buffer();
        const int tid        = threadIdx.x;
        const int stride     = blockDim.x;

        // Each thread loads elements cooperatively
        for (int i = tid; i < elements; i += stride) {
            // Use MUBUF instruction with LDS bit set
            asm volatile("buffer_load_dword %0, %1, s[0:3], 0 offen lds\n\t"
                         :
                         : "v"(&lds_buffer[i]), "v"(&global_addr[i])
                         : "memory");
        }
    }

    // Synchronize and swap buffers
    __device__ __forceinline__ void sync_and_swap() {
        // Wait for LDS prefetch to complete
        wait_for_lds();

        // Synchronize all threads in workgroup
        workgroup_barrier();

        // Swap buffers
        current_is_a = !current_is_a;
    }

    // Initial prefetch before main loop
    __device__ __forceinline__ void initial_prefetch(const T * __restrict__ global_addr, int elements) {
        prefetch_async(global_addr, elements);
        wait_for_lds();
        workgroup_barrier();
    }
};

// ================================
// Optimized Memory Copy Kernels
// ================================

// High-bandwidth memory copy using DWORDX4 with correct GCN inline assembly
inline __global__ void memcpy_dwordx4_kernel(float * __restrict__ dst, const float * __restrict__ src, size_t n_float4) {
    const int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    // Each thread handles one float4 (128 bits)
    for (size_t i = tid; i < n_float4; i += stride) {
        // Calculate per-thread offset in bytes (must be 32-bit for GCN addressing mode)
        uint32_t offset = (uint32_t)(i * 16);  // 16 bytes per float4
        
        // Temporary storage for the data
        float4 vdata;
        
        // Correct GCN inline assembly using SGPR base + VGPR offset addressing
        asm volatile(
            // Load 4 dwords from source memory
            // VDST: %0 (vdata), VOFFSET: %1 (offset), SBASE: %2 (src pointer)
            "global_load_dwordx4 %0, %1, %2\n\t"
            
            // Wait for the load to complete before storing
            "s_waitcnt vmcnt(0)\n\t"
            
            // Store 4 dwords to destination memory
            // Note: For stores, the offset comes before the data
            // VOFFSET: %1 (offset), VDATA: %0 (vdata), SBASE: %3 (dst pointer)
            "global_store_dwordx4 %1, %0, %3\n\t"
            
            // Wait for the store to complete
            "s_waitcnt vmcnt(0)"
            
            // Output operands
            : "=v"(vdata)      // %0: output, vector register for loaded data
            
            // Input operands
            : "v"(offset),     // %1: input, vector register for the offset
              "s"(src),        // %2: input, SCALAR register pair for the src pointer
              "s"(dst)         // %3: input, SCALAR register pair for the dst pointer
            
            : "memory"
        );
    }
}

// ================================
// Prefetch Instructions
// ================================

// L2 cache prefetch hint
__device__ __forceinline__ void prefetch_l2_isa(const void * addr) {
    // GLC (globally coherent) and SLC (system level coherent) flags
    // help with L2 cache utilization
    asm volatile("global_load_dword v[0:0], %0, off glc slc\n\t" : : "v"(addr) : "v0", "memory");
}

// Prefetch with non-temporal hint (bypass cache)
__device__ __forceinline__ void prefetch_nt_isa(const void * addr) {
    // NT (non-temporal) flag bypasses cache for streaming data
    asm volatile("global_load_dword v[0:0], %0, off nt\n\t" : : "v"(addr) : "v0", "memory");
}

// ================================
// Memory Fence Instructions
// ================================

// Full memory fence
__device__ __forceinline__ void memory_fence_isa() {
    // S_WAITCNT waits for all memory operations
    asm volatile(
        "s_waitcnt vmcnt(0) & lgkmcnt(0)\n\t"
        "buffer_gl0_inv\n\t"
        "buffer_gl1_inv" ::
            : "memory");
}

// Vector memory fence only
__device__ __forceinline__ void vmem_fence_isa() {
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
}

// LDS memory fence only
__device__ __forceinline__ void lds_fence_isa() {
    asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
}

// ================================
// Additional GCN ISA Instructions
// ================================

// GLOBAL_STORE_DWORD - Store 32-bit value (direct address)
__device__ __forceinline__ void global_store_dword(float * __restrict__ addr, float value) {
    // Use regular store - compiler will optimize to global_store_dword
    *addr = value;
    __builtin_amdgcn_s_waitcnt(0x3F70); // vmcnt(0)
}

// GLOBAL_STORE_DWORD with base+offset addressing using inline assembly
__device__ __forceinline__ void global_store_dword_offset(float * __restrict__ base, uint32_t offset, float value) {
    asm volatile(
        "global_store_dword %0, %1, %2\n\t"
        "s_waitcnt vmcnt(0)"
        :
        : "v"(offset),           // Input: 32-bit offset in VGPR
          "v"(value),            // Input: data in VGPR
          "s"(base)              // Input: base pointer in SGPR pair
        : "memory");
}

// GLOBAL_STORE_DWORDX2 - Store 64-bit value (direct address)
__device__ __forceinline__ void global_store_dwordx2(float2 * __restrict__ addr, float2 value) {
    // Use regular store - compiler will optimize to global_store_dwordx2
    *addr = value;
    __builtin_amdgcn_s_waitcnt(0x3F70); // vmcnt(0)
}

// GLOBAL_STORE_DWORDX2 with base+offset addressing using inline assembly
__device__ __forceinline__ void global_store_dwordx2_offset(float2 * __restrict__ base, uint32_t offset, float2 value) {
    asm volatile(
        "global_store_dwordx2 %0, %1, %2\n\t"
        "s_waitcnt vmcnt(0)"
        :
        : "v"(offset),           // Input: 32-bit offset in VGPR
          "v"(value),            // Input: data in VGPR
          "s"(base)              // Input: base pointer in SGPR pair
        : "memory");
}

// GLOBAL_STORE_DWORDX4 - Store 128-bit value (direct address)
__device__ __forceinline__ void global_store_dwordx4(float4 * __restrict__ addr, float4 value) {
    // Use regular store - compiler will optimize to global_store_dwordx4
    *addr = value;
    __builtin_amdgcn_s_waitcnt(0x3F70); // vmcnt(0)
}

// GLOBAL_STORE_DWORDX4 with base+offset addressing using inline assembly
__device__ __forceinline__ void global_store_dwordx4_offset(float4 * __restrict__ base, uint32_t offset, float4 value) {
    asm volatile(
        // For stores, offset comes before data
        "global_store_dwordx4 %0, %1, %2\n\t"
        "s_waitcnt vmcnt(0)"
        :
        : "v"(offset),           // Input: 32-bit offset in VGPR
          "v"(value),            // Input: data in VGPR
          "s"(base)              // Input: base pointer in SGPR pair
        : "memory");
}

// V_DOT4_I32_I8 - GFX906 specific dot product instruction
// Performs 4x int8 dot product: result = a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3]
__device__ __forceinline__ int32_t v_dot4_i32_i8(int32_t a, int32_t b, int32_t c) {
    int32_t result;
    asm volatile(
        "v_dot4_i32_i8 %0, %1, %2, %3\n\t"
        : "=v"(result)
        : "v"(a), "v"(b), "v"(c));
    return result;
}

// V_DOT2_F32_F16 - GFX906 specific dot product for half precision
__device__ __forceinline__ float v_dot2_f32_f16(uint32_t a, uint32_t b, float c) {
    float result;
    asm volatile(
        "v_dot2_f32_f16 %0, %1, %2, %3\n\t"
        : "=v"(result)
        : "v"(a), "v"(b), "v"(c));
    return result;
}

// S_MEMTIME - Read system timestamp
__device__ __forceinline__ uint64_t s_memtime() {
    uint64_t time;
    asm volatile(
        "s_memtime %0\n\t"
        "s_waitcnt lgkmcnt(0)"
        : "=s"(time));
    return time;
}

// Memory fence for global memory
__device__ __forceinline__ void global_memory_fence() {
    asm volatile(
        "s_waitcnt vmcnt(0)\n\t"
        "buffer_wbinvl1_vol\n\t"
        ::: "memory");
}

}  // namespace memory_isa
}  // namespace gfx906

#endif  // GGML_HIP_GFX906_OPTIMIZED
