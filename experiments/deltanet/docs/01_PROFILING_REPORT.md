# GFX906 Kernel Profiling Report - Qwen3-Next-80B-A3B

**Date:** December 7, 2025
**Model:** Qwen3-Next-80B-A3B-Instruct-Q2_K.gguf
**GPU:** AMD Instinct MI50 (gfx906)
**Test:** Quick benchmark (pp=1024, tg=128)
**Total GPU Time:** 6,074.52 ms

---

## Executive Summary

The Qwen3-Next model uses a **hybrid architecture** combining:
1. **Standard Transformer attention** (12 layers with FlashAttention/TILE kernel)
2. **Delta-Net linear attention** (36 recurrent layers with state-space mechanisms)
3. **MoE (Mixture of Experts)** FFN layers with 512 experts, 10 active per token

**90%+ of GPU time** is spent in rocBLAS GEMM kernels, with one critical issue: **22% of time uses ISA000 (generic fallback)** instead of optimized ISA906 kernels.

---

## Part 1: Qwen3-Next Special Attention Kernels

### 1.1 Architecture Overview

Qwen3-Next is a **Mamba-2 style hybrid model** with:

| Component | Value |
|-----------|-------|
| Total Layers | 48 |
| Attention Layers | 12 (standard transformer) |
| Linear Attention Layers | 36 (Delta-Net recurrent) |
| Hidden Size | 2048 |
| Attention Heads | 16 (2 KV heads, GQA ratio 8:1) |
| Head Dimension | 256 (K and V) |
| SSM State Size | 128 |
| SSM Inner Size | 4096 |
| SSM Groups | 16 |
| Experts | 512 total, 10 active |

### 1.2 Delta-Net Linear Attention (36 layers)

The Delta-Net attention is a **recurrent linear attention mechanism** that replaces standard softmax attention with a state-space approach. Key characteristics:

**Mathematical Operations:**
```
1. L2-normalize Q, K
2. Scale Q by 1/sqrt(S_v)
3. Apply sigmoid to beta (forgetting factor)
4. Compute decay_mask = exp(cumsum(gate) * causal_mask)
5. Solve triangular system: (I - tril(attn)) * X = attn
6. Update recurrent state: state = state * exp(g_last) + k @ v_new
```

**GPU Kernel Implications:**
- Heavy use of **element-wise operations**: exp, sigmoid, cumsum, mul, add, sub, neg
- **Triangular solve** (`ggml_solve_tri`) - custom kernel
- **Multiple small matrix multiplies** for attention computation
- **State updates** requiring memory bandwidth

**Two Execution Paths:**
1. **Chunked mode** (`build_delta_net_chunking`): Used for sequences > 64 tokens (CHUNK_SIZE)
   - Processes in 64-token chunks
   - More GEMM calls but better parallelism

2. **Recurrent mode** (`build_delta_net_recurrent`): Used for sequences ≤ 64 tokens
   - Single-pass computation
   - Fewer but larger operations

### 1.3 Standard Transformer Attention (12 layers)

Uses **TILE FlashAttention kernel** (seen in output: `Flash Attention kernel: TILE`)

**Operations:**
- Gated Q projection (Q + gate split)
- Q/K L2 normalization
- RoPE positional encoding
- Scaled dot-product attention with FlashAttention
- Sigmoid gating on output

### 1.4 SSM Convolution

Each Delta-Net layer includes a **1D convolution** on Q, K, V:
- Kernel size: 4 (from `hparams.ssm_d_conv`)
- Applied via `ggml_ssm_conv` kernel
- Followed by SiLU activation

---

## Part 2: GEMM Launch Analysis

### 2.1 Kernel Call Statistics

| Rank | Kernel | Calls | Time (ms) | % | ISA |
|------|--------|-------|-----------|---|-----|
| 1 | MT64x64x16 | 294,912 | 1,391.04 | 22.9% | **906** |
| 2 | MT16x16x8 | 147,456 | 1,333.15 | 21.9% | **000** ⚠️ |
| 3 | MT64x64x16 | 110,592 | 1,266.66 | 20.9% | **906** |
| 4 | MT64x64x16 | 184,320 | 886.23 | 14.6% | **906** |
| 5 | MT64x32x16 | 147,456 | 628.86 | 10.4% | **906** |
| 8 | MT32x128x16 | 188 | 80.12 | 1.3% | **000** ⚠️ |
| 9 | MT32x128x32 | 188 | 52.88 | 0.9% | **906** |

**Total GEMM calls: ~885,000+**

### 2.2 Why So Many GEMM Launches?

**Root Causes:**

1. **Delta-Net Linear Attention (36 layers × many matmuls per layer)**
   - Each layer performs 10+ matrix multiplications:
     - `k @ k_beta` (decay computation)
     - `attn @ v_beta` (value computation)
     - `attn @ k_beta_gexp` (cumulative decay)
     - `q @ k` (attention scores)
     - `state @ k_cumdecay` (state-value interaction)
     - `q_gexp @ state` (inter-chunk attention)
     - `attn @ v_new` (final output)
     - State update multiplies
   - **Estimate: 36 layers × ~10 matmuls × ~4000 tokens = ~1.4M potential GEMM calls**

2. **MoE FFN Layers (48 layers)**
   - Each MoE layer with 10 active experts requires:
     - Router/gate computation
     - 10× up projections
     - 10× gate projections
     - 10× down projections
   - **Estimate: 48 × 30+ operations = 1440+ GEMM calls per token**

3. **Small Matrix Dimensions**
   - Delta-Net uses small matrices (state_size=128, head_dim=256)
   - These trigger small tile kernels (MT16x16x8)
   - Small GEMMs have poor GPU utilization

### 2.3 ISA000 Fallback Analysis

**Critical Finding:** The MT16x16x8 kernel uses ISA000 (generic fallback) consuming **22% of total GPU time**.

**Why ISA000?**
- rocBLAS selects kernels based on problem size
- For small matrices (M=16, N=16, K=8), there may not be an optimized ISA906 kernel
- Falls back to generic implementation

**Kernel Name Decoding (MT16x16x8):**
```
Cijk_Alik_Bljk_SB_MT16x16x8_SN_...ISA000...
       │    │      │
       │    │      └─ Macro Tile: 16x16, K-tile: 8
       │    └─ Layout: A[lik], B[ljk] (both non-transposed internally)
       └─ Contraction type
```

### 2.4 Optimization Opportunities

**A. Reduce GEMM Launches:**

1. **Kernel Fusion** - Fuse consecutive operations:
   - `mul → mul → exp → mul` chains in Delta-Net could be fused
   - State update operations could be combined

2. **Batched GEMM** - Instead of many small GEMMs:
   ```cpp
   // Current: 36 separate calls
   for (layer : layers) gemm(q, k)

   // Better: 1 batched call
   batched_gemm(Q_all, K_all, batch_count=36)
   ```

3. **Chunking Strategy** - Current CHUNK_SIZE=64 may not be optimal for MI50
   - Larger chunks = fewer GEMM calls
   - But memory constraints limit chunk size

**B. Fix ISA000 Fallback:**

1. **Custom Small GEMM Kernel** - Write hand-tuned kernel for 16x16x8 tiles
2. **Pad to Larger Tiles** - Pad matrices to 32x32 or 64x64 for better kernel selection
3. **Use Different Algorithm** - For very small matrices, use direct computation instead of GEMM

**C. Algorithmic Changes:**

1. **Fused Delta-Net Kernel** - Single kernel for entire Delta-Net attention
2. **Sparse Attention** - Skip near-zero attention weights
3. **Quantized Attention** - Use INT8 for attention computation

---

## Part 3: Complete Kernel Profile

### 3.1 All Kernels by Time

| Rank | Kernel | Calls | Time (ms) | % | Category |
|------|--------|-------|-----------|---|----------|
| 1 | rocBLAS_gemm<MT64x64x16_ISA906> | 294,912 | 1,391.04 | 22.90% | GEMM |
| 2 | rocBLAS_gemm<MT16x16x8_ISA000> | 147,456 | 1,333.15 | 21.95% | GEMM ⚠️ |
| 3 | rocBLAS_gemm<MT64x64x16_ISA906> | 110,592 | 1,266.66 | 20.85% | GEMM |
| 4 | rocBLAS_gemm<MT64x64x16_ISA906> | 184,320 | 886.23 | 14.59% | GEMM |
| 5 | rocBLAS_gemm<MT64x32x16_ISA906> | 147,456 | 628.86 | 10.35% | GEMM |
| 6 | concat_f32_dim0 | 18,864 | 230.78 | 3.80% | Memory |
| 7 | concat_f32_dim1 | 32,256 | 147.98 | 2.44% | Memory |
| 8 | rocBLAS_gemm<MT32x128x16_ISA000> | 188 | 80.12 | 1.32% | GEMM ⚠️ |
| 9 | rocBLAS_gemm<MT32x128x32_ISA906> | 188 | 52.88 | 0.87% | GEMM |
| 10 | pad_f32 | 720 | 30.41 | 0.50% | Memory |
| 11 | scale_f32 | 5,036 | 24.94 | 0.41% | Elementwise |
| 12 | k_get_rows_float | 133 | 0.75 | 0.01% | Memory |
| 13 | tri_kernel | 133 | 0.69 | 0.01% | Special |
| 14 | scale_f32 (variant) | 4 | 0.03 | 0.00% | Elementwise |

### 3.2 Kernel Categories Summary

| Category | Time (ms) | % | Kernels |
|----------|-----------|---|---------|
| **GEMM (ISA906)** | 4,225.67 | 69.56% | 5 variants |
| **GEMM (ISA000)** | 1,413.27 | 23.27% | 2 variants ⚠️ |
| **Memory Ops** | 409.92 | 6.75% | concat, pad, get_rows |
| **Elementwise** | 24.97 | 0.41% | scale |
| **Special** | 0.69 | 0.01% | tri_kernel |

### 3.3 Detailed Kernel Analysis

#### GEMM Kernels

**MT64x64x16_ISA906** (3 variants, 58.3% total)
- Optimal tile size for medium matrices
- Used for: MoE expert projections, attention projections
- Well-optimized for gfx906

**MT64x32x16_ISA906** (10.4%)
- Rectangular tile for non-square matrices
- Used for: Down projections, narrow matrices

**MT16x16x8_ISA000** (21.9%) ⚠️ **CRITICAL**
- Generic fallback kernel
- Very small tile = poor efficiency
- **Source:** Small Delta-Net state multiplications (128x128, 256x128, etc.)
- **Action Required:** Custom kernel or padding strategy

**MT32x128x16_ISA000** (1.3%) ⚠️
- Another fallback kernel
- Used for: Certain attention dimensions

#### Memory Kernels

**concat_f32_dim0/dim1** (6.2% combined, 51,120 calls)
- Heavy use in Delta-Net for state concatenation
- `concat(output, state, dim=0)` in every linear attention layer
- Optimization: Could be fused with preceding operation

**pad_f32** (0.5%, 720 calls)
- Padding for Delta-Net chunking (pad to CHUNK_SIZE=64)
- 720 calls / 36 layers = 20 pads per layer (Q, K, V, g, beta × 4 dimensions?)

#### Special Kernels

**tri_kernel** (133 calls)
- Triangular matrix operations
- Used for: `ggml_tri` (causal mask), `ggml_solve_tri` (linear system solve)
- Part of Delta-Net's unique triangular system solve

---

## Part 4: Recommendations

### Immediate Actions

1. **Investigate MT16x16x8 ISA000 Usage**
   - Profile which operations trigger this kernel
   - Consider padding small matrices to 32x32 minimum

2. **Reduce concat Operations**
   - 51,000+ concat calls is excessive
   - Potential for output buffer pre-allocation

3. **Custom Delta-Net Kernel**
   - Fuse the entire Delta-Net attention into one kernel
   - Would eliminate most small GEMM calls

### Medium-Term Optimizations

1. **Batched GEMM for Delta-Net**
   - Batch similar-sized operations across layers
   - Reduce kernel launch overhead

2. **Memory Layout Optimization**
   - Current layout may cause unnecessary transposes
   - Profile memory bandwidth utilization

3. **Tune CHUNK_SIZE**
   - Current value (64) may not be optimal for MI50
   - Test 128, 256 chunk sizes

### Long-Term Architecture Changes

1. **Custom ISA906 Small GEMM**
   - Write hand-tuned assembly for 16x16 tiles
   - Target the 22% ISA000 time

2. **Sparse Delta-Net**
   - Many attention weights are near-zero
   - Sparse computation could save significant time

3. **Quantized Linear Attention**
   - INT8 state representation
   - Would reduce memory bandwidth bottleneck

---

## Appendix A: rocBLAS Kernel Name Decoder

```
Cijk_Alik_Bljk_SB_MT64x64x16_SN_..._ISA906_...
│     │    │    │  │          │        │
│     │    │    │  │          │        └─ ISA: 906=gfx906 optimized, 000=generic
│     │    │    │  │          └─ SN/SE: Split N/Split E (tiling strategy)
│     │    │    │  └─ MacroTile: M×N×K tile dimensions
│     │    │    └─ SB: Strided Batched
│     │    └─ B layout: [ljk] = column-major
│     └─ A layout: [lik] = row-major
└─ Contraction type (GEMM)
```

---

## Appendix B: Qwen3-Next Model Dimensions

Key tensor dimensions affecting GEMM sizes:

| Operation | M | N | K | Notes |
|-----------|---|---|---|-------|
| QKV Projection | 2048 | tokens | 2048 | Main projections |
| MoE Up | 512 | tokens | 2048 | Per-expert |
| MoE Gate | 512 | tokens | 2048 | Per-expert |
| MoE Down | 2048 | tokens | 512 | Per-expert |
| Delta-Net k@k_beta | 128 | 128 | tokens | State interaction |
| Delta-Net state | 128 | 128 | 32 | State update |
| FlashAttn QK | heads×256 | seq | 256 | Attention |

---

*Report generated by kernel discovery profiler*
*Fix applied: rocBLAS direct calls for gfx906 (bypassing hipBLASLt)*
