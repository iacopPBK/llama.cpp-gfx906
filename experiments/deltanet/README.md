# Delta-Net Fused Kernel for AMD MI50 (gfx906)

Fused GPU kernel implementation for Delta-Net linear attention, targeting Qwen3-Next hybrid models on AMD MI50.

## Project Status

```
Phase 0-8: Kernel Validation    [████████████████████] 100% (26/26 tests passing)
Kernel Fusion:                  [░░░░░░░░░░░░░░░░░░░░]   0%
llama.cpp Integration:          [░░░░░░░░░░░░░░░░░░░░]   0%
```

## Quick Start

```bash
# Build
make

# Run all tests
HSA_OVERRIDE_GFX_VERSION=9.0.6 HIP_VISIBLE_DEVICES=0 ./test_deltanet

# Run specific phase
./test_deltanet --phase 1   # Element-wise ops
./test_deltanet --phase 8   # Full Delta-Net

# List available tests
./test_deltanet --list
```

## Directory Structure

```
experiments/deltanet/
├── README.md              # This file
├── Makefile               # Build system
├── reference.h            # Host reference API
├── reference.cpp          # Host reference implementations (GOLDEN)
├── test_harness.cpp       # GPU kernels + validation tests
├── test_deltanet          # Built executable
└── docs/
    ├── 01_PROFILING_REPORT.md      # Initial profiling (885K+ GEMM calls)
    ├── 02_ALGORITHM_ANALYSIS.md    # Delta-Net math (WY representation)
    ├── 03_IMPLEMENTATION_PLAN.md   # 8-phase development plan
    └── 04_TECHNICAL_REFERENCE.md   # Complete kernel documentation
```

## Documentation

| Document | Description |
|----------|-------------|
| [01_PROFILING_REPORT](docs/01_PROFILING_REPORT.md) | Kernel profiling that identified the performance problem |
| [02_ALGORITHM_ANALYSIS](docs/02_ALGORITHM_ANALYSIS.md) | Deep dive into Delta-Net algorithm and WY representation |
| [03_IMPLEMENTATION_PLAN](docs/03_IMPLEMENTATION_PLAN.md) | 8-phase incremental development plan |
| [04_TECHNICAL_REFERENCE](docs/04_TECHNICAL_REFERENCE.md) | Complete kernel reference with code examples |

## Test Phases

| Phase | Description | Tests | Status |
|-------|-------------|-------|--------|
| 1 | Element-wise (sigmoid, exp, mul, cumsum) | 4 | ✅ |
| 2 | Triangular (tril, causal, eye, solve_tri) | 5 | ✅ |
| 3 | Matrix ops (GEMM NN/NT/TN, tiled) | 5 | ✅ |
| 4 | Decay mask computation | 2 | ✅ |
| 5 | Attention matrix construction | 2 | ✅ |
| 6 | Intra-chunk computation | 4 | ✅ |
| 7 | State interaction | 2 | ✅ |
| 8 | Full multi-chunk Delta-Net | 2 | ✅ |

## The Problem

Qwen3-Next's 36 Delta-Net layers generate **885,000+ small GEMM kernel launches** per inference, causing:
- High kernel launch overhead
- Poor GPU utilization
- 22% of time in ISA000 fallback (generic) instead of ISA906 (optimized)

## The Solution

Fuse all Delta-Net operations into a **single kernel** per chunk:
- Eliminate kernel launch overhead
- Keep data in LDS/registers
- Enable operation fusion and pipelining

## Key Dimensions (Qwen3-Next)

```cpp
S_K = S_V = 128      // State/head dimension
CHUNK_SIZE = 64      // Tokens per chunk
NUM_K_HEADS = 16     // Key heads
NUM_V_HEADS = 32     // Value heads
```

## Next Steps

1. **Parallel Triangular Solve** - Replace sequential 1-thread solver with wave-parallel version
2. **Kernel Fusion** - Combine 16+ separate kernels into 1 fused kernel
3. **Multi-head Batching** - Process all heads in parallel
4. **llama.cpp Integration** - Add GGML_OP_DELTANET operator

## Hardware Target

- **GPU**: AMD Instinct MI50 (gfx906)
- **LDS**: 64 KB per workgroup
- **Wavefront**: 64 threads
- **Constraints**: No on-device printf, no tensor cores

## Validation Strategy

Since gfx906 doesn't support on-device printf:
1. All kernels validated against host C++ reference
2. Intermediate results copied to host for comparison
3. Max absolute difference reported for each test
4. Tolerance: 1e-4 to 1e-6 depending on operation

## License

Part of llama.cpp - MIT License
