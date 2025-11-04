# llama.cpp-gfx906: AMD MI50/MI60/Vega7 Optimized Fork

This fork implements **low-level AMD GCN ISA optimizations** for llama.cpp inference, specifically targeting the AMD MI50/MI60/Vega VII GPUs (GFX906 / Vega 20 architecture).

**Key Achievement**: Replaced generic shuffle-based reductions with **fused DPP+ALU instructions**, reducing instruction count by ~37% in critical reduction paths.

---

## Performance Improvements

**Test Configuration:** ROCm backend, ngl=99, threads=12, batch=1024, KV cache: q8_0, Flash Attention enabled

| Model | Quant | Context | Test Type | Vanilla (t/s) | Fork (t/s) | Improvement | Speedup |
|-------|-------|---------|-----------|---------------|------------|-------------|---------|
| **Qwen3 4B** | Q4_0 | d=0 | pp512 | 1782.62 ± 0.59 | 2023.40 ± 0.86 | +240.78 | +13.5% |
| | | | tg128 | 127.95 ± 0.02 | 134.61 ± 0.04 | +6.66 | +5.2% |
| | | d=2048 | pp512 | 1382.44 ± 17.69 | 1612.72 ± 0.95 | +230.28 | +16.7% |
| | | | tg128 | 81.58 ± 1.56 | 107.05 ± 0.03 | +25.47 | **+31.2%** |
| **Qwen3 4B** | Q4_1 | d=0 | pp512 | 1859.20 ± 0.61 | 1921.99 ± 0.46 | +62.79 | +3.4% |
| | | | tg128 | 132.30 ± 0.01 | 139.82 ± 0.02 | +7.52 | +5.7% |
| | | d=2048 | pp512 | 1498.35 ± 0.51 | 1541.53 ± 1.72 | +43.18 | +2.9% |
| | | | tg128 | 88.51 ± 0.01 | 110.83 ± 0.02 | +22.32 | +25.2% |
| **Qwen3VLMoE 30B** | Q4_1 | d=0 | pp512 | 1245.10 ± 11.10 | 1362.27 ± 11.47 | +117.17 | +9.4% |
| | | | tg128 | 97.65 ± 0.04 | 100.87 ± 0.03 | +3.22 | +3.3% |
| | | d=2048 | pp512 | 1022.02 ± 19.17 | 1146.50 ± 8.23 | +124.48 | +12.2% |
| | | | tg128 | 70.10 ± 0.69 | 81.86 ± 0.04 | +11.76 | +16.8% |


**Legend:**
- **pp512:** Prompt processing with 512 tokens
- **tg128:** Text generation with 128 tokens
- **d=0:** No context
- **d=2048:** With 2048 tokens of context
- **t/s:** Tokens per second

---

## Key Optimizations

### 1. Fused DPP Instructions (Main Optimization)

Replaced separate shuffle + arithmetic operations with single fused DPP+ALU instructions:

**Before (2 instructions):**
```cpp
x = __shfl_xor(x, 1);  // DPP shuffle
x = x + other;          // ALU add
```

**After (1 instruction):**
```cpp
x = hip_add_xor1_f32(x);  // Fused v_add_f32_dpp
```

**Impact:**
- Reduction operations: **37% fewer instructions** (10-15 → 8 instructions)
- Fused XOR 1, 2, 8 patterns
- XOR 4, 16 remain unfused (architectural limitation)
- Applied to: argmax, quantization, flash attention, top-k MoE

### 2. Vectorized Q4_0/Q4_1 Memory Loads

Replaced 8 scalar 32-bit loads with 2 vectorized 128-bit int4 loads:

**Impact:**
- ~2× memory throughput for Q4_0/Q4_1 quantization formats
- Improved vec_dot operation performance

### 3. Flash Attention Fixes

- Fixed kernel selection logic for GFX906 single-token generation
- GCN-tuned thread counts (`nthreads_KQ_q=2`, `nthreads_V_q=4`)
- Generic reduction templates with type-aware dispatch
- Restored GFX906 compatibility

### 4. DPP Architecture Optimizations

**DPP (Data Parallel Primitives)** on AMD GCN allow:
- Lane-to-lane data movement within a wavefront (64 threads)
- **Fusion with ALU operations** (add, max, etc.)
- Single-cycle execution for common patterns

**Barrier management** (critical for correctness):
```cpp
asm volatile(
    "s_nop 4\n"  // FIRST DPP: EXEC mask hazard protection
    "v_add_f32_dpp %0, %1, %1 quad_perm:[1,0,3,2] ..."
    : "=v"(result) : "v"(x)
);

asm volatile(
    "s_nop 1\n"  // SUBSEQUENT DPP: VGPR→DPP data hazard
    "v_add_f32_dpp %0, %1, %1 quad_perm:[2,3,0,1] ..."
    : "=v"(result) : "v"(x)
);
```



---

## Tested Models

All llama.cpp supported models work with this fork. Extensively tested with:
- **Qwen3-4B** (Q4_0, Q4_1)
- **Qwen3VLMoE-30B** (Q4_0, Q4_1) - Vision + MoE model

---

## Quick Start

### Prerequisites

- **ROCm 7.0.1** (tested version)
- **CMake 3.21+**
- **HIP compiler toolchain**
- **AMD GFX906 GPU** (MI50/MI60/Vega VII)
- **Ubuntu 24.04** (tested, other distros should work)

### System Dependencies

```bash
# Ubuntu
sudo apt update
sudo apt install cmake build-essential

# Install ROCm 7.0.1 following AMD's official guide
# Note: Tensile library for gfx906 must be imported for ROCm 7.0.1

# Verify ROCm installation
/opt/rocm/bin/rocm-smi
```

### Build Instructions

#### 1. Clone the repository

```bash
git clone https://github.com/iacopPBK/llama.cpp-gfx906.git
cd llama.cpp-gfx906
```

#### 2. Compile using the provided script

```bash
chmod +x SCRIPT_compile_MI50.sh
./SCRIPT_compile_MI50.sh
```

The compilation script automatically:
- Sets GFX906-specific compiler flags
- Enables HIP backend with GFX906 optimizations
- Builds with flash attention support
- Links against ROCm libraries (rocBLAS, hipBLAS)

#### 3. Launch the server

```bash
# Edit SCRIPT_launch_server_MI50.sh to set your model path
vim SCRIPT_launch_server_MI50.sh

# Launch server with Flash Attention and KV quantization
./SCRIPT_launch_server_MI50.sh
```

### Environment Variables

The optimized build sets these automatically:

```bash
export HSA_OVERRIDE_GFX_VERSION=9.0.6
export HIP_VISIBLE_DEVICES=0
export ROCR_VISIBLE_DEVICES=0
export GGML_BACKEND_HIP=1
export HCC_AMDGPU_TARGET=gfx906
```

---

## Build Configuration

The build enables these optimizations:

- `GGML_HIP=ON` - Enable HIP backend
- `GGML_HIP_GFX906_OPTIMIZED=ON` - GFX906-specific optimizations
- `CMAKE_HIP_ARCHITECTURES=gfx906` - Target GFX906 architecture
- Flash attention with F16 precision (hardcoded)

---

## Benchmarking

### Run llama-bench

```bash
./SCRIPT_llama_bench.sh
```

### Vision Models

```bash
./build/bin/llama-cli \
  -m model.gguf \
  --image test.jpg \
  -p "Describe this image"
```

## Technical Details

### Files Modified

**Core Optimization Files:**
1. `ggml/src/ggml-cuda/common.cuh` - Unified DPP optimization section, fused DPP+ALU templates
2. `ggml/src/ggml-cuda/mmq.cuh` - Vectorized int4 loads for Q4_0/Q4_1
3. `ggml/src/ggml-cuda/quantize.cu` - Fused warp reductions
4. `ggml/src/ggml-cuda/fattn-vec.cuh` - GCN-tuned thread counts
5. `ggml/src/ggml-cuda/fattn.cu` - Fixed kernel selection
6. `ggml/src/ggml-cuda/vecdotq.cuh` - 2-byte aligned loads
7. `ggml/src/ggml-cuda/argmax.cu` - Fused reductions
8. `ggml/src/ggml-cuda/topk-moe.cu` - Fused reductions



---

## References

- [AMD GCN Architecture Documentation](https://gpuopen.com/learn/amdgcn-assembly/)
- [llama.cpp upstream](https://github.com/ggml-org/llama.cpp)
- [AMD ROCm Documentation](https://rocm.docs.amd.com/)

---

*Built with care for the AMD GFX906 community ❤️‍🔥*
