#pragma once

// GFX906 FlashAttention Helper Functions
// Contains AMD-specific optimizations for FlashAttention kernel selection

#if defined(GGML_USE_HIP)

#include <cstdint>

struct ggml_tensor;

// Check if Q8 Flash Attention can be used for given tensor configuration
// 
// Q8 Flash Attention is optimized for specific head sizes and requires
// K or V tensors to be in Q8_0 format.
//
// Parameters:
//   - K: Key tensor
//   - V: Value tensor
//
// Returns:
//   - true if Q8 FA can be used, false otherwise

static inline bool gfx906_fattn_can_use_q8(const ggml_tensor * K, const ggml_tensor * V) {
    // Check if either K or V is Q8_0 quantized
    if (K->type != GGML_TYPE_Q8_0 && V->type != GGML_TYPE_Q8_0) {
        return false;
    }
    
    // Check head size constraints
    // Q8 FA supports head sizes divisible by 32, except specific excluded sizes
    const int64_t head_size = K->ne[0];
    const bool head_size_supported = (head_size % 32 == 0) &&
                                     (head_size != 40) &&
                                     (head_size != 80) &&
                                     (head_size != 112) &&
                                     (head_size != 576);
    
    return head_size_supported;
}

#endif // GGML_USE_HIP
