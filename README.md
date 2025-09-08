# llama.cpp-gfx906: AMD MI50/MI60/Vega7 fork

**Specialized llama.cpp fork with GFX906 flash attention optimizations for D=128 head dimension models ONLY!**

This fork is specifically optimized for AMD GFX906 architecture (MI50, MI60, Vega VII) and targets models with **head dimension D=128** (such as Qwen3-30B series). The aim of this fork is to be able to run a QWEN30B session with 32K ctx on a single card without losing too much speed. For this reason the fork won't work with smaller models (you can check the huggingface model sheet for key and value lengths).

---

## Acknowledgments

**Special thanks to [skyne98](https://github.com/skyne98/ggml-gfx906)** for the foundational work, and of course to the whole **[ggml-org](https://github.com/ggml-org/llama.cpp)** open source community:

- **All GFX906 primitive operations** (`gfx906-wave-primitives*.cuh`)
- **GEMM kernel implementations** (`gemm-gfx906*.cu/cuh`) 
- **Memory access patterns** (`gfx906-memory-*.cuh`)
- **Assembly optimizations** (`gfx906-asm-*`)
- **Auto-tuning framework** (`gemm-gfx906-autotuner.cuh`)

Thanks to all the https://discord.gg/sgjdAU9eRC people for the efforts on gfx906 optimization.

**This fork builds upon skyne98's GFX906 optimization work, i did focus specifically on flash attention improvements for D=128 models.**

---

## Key Features of fattn-tile-f16-gfx906.cu

- **Native 64-thread wavefront support** (vs 32-thread warps)
- **Register blocking optimization** (reduction in shared memory accesses)  
- **V_DOT2_F32_F16 native instruction usage** for dual-FP16 operations
- **Strategic bank conflict elimination** with optimized padding
- **Forced F16 precision** for flash attention operations
- **Optimized for D=128 head dimension** with runtime validation

---

## Target Hardware & Models

### Supported GPUs
- **AMD MI50** (Vega 20) (only one actually tested)
- **AMD MI60** (Vega 20) 
- **AMD Vega VII** (Vega 20)

### Supported Models
- **Models with D=128 head dimension only** it will crash with a message for other D dimensions.
- Tested extensively with **Qwen3-30B-A3B series** (Q4_0, Q4_1)
- Compatible with models using similar attention architecture (needs testing!)

---

## Performance Improvements

Benchmarks on **Qwen3-30B-A3B-Thinking-2507-Q4_0** with **AMD MI50**:

### Prompt Processing (tokens/second) - llama-bench results

**Device 0:** AMD Instinct MI50/MI60, gfx906:sramecc+:xnack- (0x906), VMM: no, Wave Size: 64

#### With KV Quantization (Q8_0)

| Model | Size | Params | Backend | ngl | threads | n_batch | type_k | type_v | fa | Test | t/s |
|-------|------|--------|---------|-----|---------|---------|--------|--------|----|------|-----|
| qwen3moe 30B.A3B Q4 | 16.18 GiB | 30.53 B | ROCm | 99 | 12 | 1024 | q8_0 | q8_0 | 1 | pp512 | 1224.07 ± 6.93 |
| qwen3moe 30B.A3B Q4 | 16.18 GiB | 30.53 B | ROCm | 99 | 12 | 1024 | q8_0 | q8_0 | 1 | pp1024 | 1168.62 ± 5.28 |
| qwen3moe 30B.A3B Q4 | 16.18 GiB | 30.53 B | ROCm | 99 | 12 | 1024 | q8_0 | q8_0 | 1 | pp2048 | 1049.93 ± 1.75 |
| qwen3moe 30B.A3B Q4 | 16.18 GiB | 30.53 B | ROCm | 99 | 12 | 1024 | q8_0 | q8_0 | 1 | pp4096 | 861.60 ± 1.48 |

#### Without KV Quantization

| Model | Size | Params | Backend | ngl | threads | n_batch | Test | t/s |
|-------|------|--------|---------|-----|---------|---------|------|-----|
| qwen3moe 30B.A3B Q4 | 16.18 GiB | 30.53 B | ROCm | 99 | 12 | 1024 | pp512 | 1167.28 ± 8.12 |
| qwen3moe 30B.A3B Q4 | 16.18 GiB | 30.53 B | ROCm | 99 | 12 | 1024 | pp1024 | 1084.71 ± 5.40 |
| qwen3moe 30B.A3B Q4 | 16.18 GiB | 30.53 B | ROCm | 99 | 12 | 1024 | pp2048 | 942.85 ± 1.64 |
| qwen3moe 30B.A3B Q4 | 16.18 GiB | 30.53 B | ROCm | 99 | 12 | 1024 | pp4096 | 773.98 ± 2.30 |

### Generation Speed (ms per token) - llama-bench results

#### With KV Quantization (Q8_0)

| Model | Size | Params | Backend | ngl | threads | n_batch | type_k | type_v | fa | Test | t/s |
|-------|------|--------|---------|-----|---------|---------|--------|--------|----|------|-----|
| qwen3moe 30B.A3B Q4 | 16.18 GiB | 30.53 B | ROCm | 99 | 12 | 1024 | q8_0 | q8_0 | 1 | tg128 | 63.00 ± 0.07 |
| qwen3moe 30B.A3B Q4 | 16.18 GiB | 30.53 B | ROCm | 99 | 12 | 1024 | q8_0 | q8_0 | 1 | tg256 | 62.83 ± 0.12 |

#### Without KV Quantization

| Model | Size | Params | Backend | ngl | threads | n_batch | Test | t/s |
|-------|------|--------|---------|-----|---------|---------|------|-----|
| qwen3moe 30B.A3B Q4_0 | 16.18 GiB | 30.53 B | ROCm | 99 | 12 | 1024 | tg128 | 79.92 ± 0.23 |
| qwen3moe 30B.A3B Q4_0 | 16.18 GiB | 30.53 B | ROCm | 99 | 12 | 1024 | tg256 | 77.87 ± 0.18 |

---

## Quick Start

### Prerequisites

- **ROCm 6.4.1** (tested version - other versions may work)
- **CMake 3.21+**
- **HIP compiler toolchain**
- **AMD GFX906 GPU** (MI50/MI60/Vega VII)
- **UBUNTU 24.04** (should work with other systems, not tested)

### System Dependencies

```bash
# Ubuntu
sudo apt update
sudo apt install cmake build-essential

# Install ROCm 6.4.1 following AMD's official guide

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

# Launch server with FA and KV quantizations
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

## Technical Details

### Memory Optimizations

- **KV Cache Padding**: +48 bytes //need to test other values!
- **Q Cache Padding**: +32 bytes //need to test other values!
- **Register Blocking**: BLOCK_SIZE=8 for memory access reduction

### Compute Optimizations  

- **64-thread wavefronts**: Native GFX906 wavefront size support
- **V_DOT2_F32_F16**: Hardware dual-FP16 dot product instructions
- **DS_SWIZZLE**: Efficient cross-SIMD unit communication
- **Scalar half operations**: Fixed numerical stability of original fattn-tile-f16 kernel

---

## Architecture Details

### GFX906-Specific Files Added

- `fattn-tile-f16-gfx906.cu` - Optimized flash attention kernel
- `gfx906-*.cuh` - GFX906 primitive operations and memory patterns
- `gemm-gfx906*.cu/cuh` - GEMM kernel optimizations
- `gfx906-asm-kernels.s` - Hand-optimized assembly kernels

### Modified Core Files

- `fattn.cu` - GFX906 detection and F16 kernel path forcing
- `common.cuh` - 64-thread wavefront reduction operations
- `vendors/hip.h` - Enabled warp sync builtins
- `CMakeLists.txt` - GFX906 build configuration

---

## Flash Attention Kernel Optimizations Explained

The `fattn-tile-f16-gfx906.cu` kernel implements several key optimizations for GFX906 architecture:

### 1. Wavefront-Aware Design

- **64-thread wavefronts**: Uses native GFX906 wavefront size instead of 32-thread CUDA warps
- **Cross-SIMD communication**: Lane XOR operations for efficient data exchange between compute units
- **Native intrinsics**: `__lane_id()` and wavefront-specific functions

### 2. Register Blocking Strategy  

```cpp
#define BLOCK_SIZE 8  // Can be tuned to explore possible performance improvements
```

- **8x memory access reduction**: Loads 8 dual-FP16 values into registers per MAC operation
- **Improved ILP**: Better instruction-level parallelism utilizing GFX906's 256 VGPR register file
- **Cache efficiency**: Reduces shared memory traffic by factor of 8

### 3. Strategic Memory Padding

```cpp 
#define GFX906_KV_PADDING 48  // Can be tuned to explore possible performance improvements
#define GFX906_Q_PADDING  32  // Can be tuned to explore possible performance improvements  
```

- **Bank conflict elimination**: Ensures different rows map to different memory banks
- **32-bank memory optimization**: Tailored for GFX906's shared memory architecture
- **D=128 specific**: Padding values optimized for 128-dimension head size

### 4. Native Instruction Usage

- **V_DOT2_F32_F16**: Hardware dual-FP16 dot product via `gfx906_dot2_f16()`
- **DS_SWIZZLE operations**: Efficient reduction operations in `wave_reduce_max()`
- **Scalar half precision**: Avoided problematic `half2` operations for numerical stability (necessary to make the f16 precision kernel to work coherently).

### 5. Architecture-Specific Launch Configuration

```cpp
__launch_bounds__(nwarps*64, 2)  // 64 threads per wavefront, 2 wavefronts per CU
```

- Designed for GFX906 compute unit structure, maximizes use of available VGPR and LDS memory (slightly lower than 64k)

The result is a flash attention kernel that achieves **5-11% performance improvement** on prompt processing tasks with respect to no flash attention, with the largest gains on longer sequences where memory access patterns matter most.

---

*Built with care for the AMD GFX906 community ❤️‍🔥 x 1000*
