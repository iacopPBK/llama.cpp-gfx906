#!/bin/bash
cat << 'EOF'

   ██╗     ██╗      █████╗ ███╗   ███╗ █████╗    ██████╗██████╗ ██████╗
   ██║     ██║     ██╔══██╗████╗ ████║██╔══██╗  ██╔════╝██╔══██╗██╔══██╗
   ██║     ██║     ███████║██╔████╔██║███████║  ██║     ██████╔╝██████╔╝
   ██║     ██║     ██╔══██║██║╚██╔╝██║██╔══██║  ██║     ██╔═══╝ ██╔═══╝
   ███████╗███████╗██║  ██║██║ ╚═╝ ██║██║  ██║  ╚██████╗██║     ██║
   ╚══════╝╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝╚═╝  ╚═╝   ╚═════╝╚═╝     ╚═╝
            ██████╗ ███████╗██╗  ██╗ █████╗  ██████╗  ██████╗
           ██╔════╝ ██╔════╝╚██╗██╔╝██╔══██╗██╔═████╗██╔════╝
           ██║  ███╗█████╗   ╚███╔╝ ╚██████║██║██╔██║███████╗
           ██║   ██║██╔══╝   ██╔██╗  ╚═══██║████╔╝██║██╔═══██╗
           ╚██████╔╝██║     ██╔╝ ██╗ █████╔╝╚██████╔╝╚██████╔╝
            ╚═════╝ ╚═╝     ╚═╝  ╚═╝ ╚════╝  ╚═════╝  ╚═════╝


EOF

export HSA_OVERRIDE_GFX_VERSION=9.0.6
export HIP_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=0
export ROCR_VISIBLE_DEVICES=0
export GGML_BACKEND_HIP=1
export HCC_AMDGPU_TARGET=gfx906
export GGML_CUDA_Q8_CACHE=1
export GGML_CUDA_RMS_MUL_MMQ_FUSION=1
export GGML_GFX906_CUBLAS_DEBUG=1                                                                              
export GGML_GFX906_MMF_DEBUG=1    
#unset GGML_CUDA_DISABLE_GRAPHS                                                                             

# Disable debug output for cleaner benchmark results
unset GGML_CUDA_Q8_CACHE_DEBUG

MODEL_PATH="/media/iacoppbk/80F42C9BF42C96061/llms/Qwen3-VL-30B-A3B-Thinking-Q4_1.gguf"
#MODEL_PATH="/media/iacoppbk/80F42C9BF42C96061/llms/openai_gpt-oss-20b-MXFP4.gguf"
MODEL_PATH="/home/iacoppbk/Downloads/GLM-4.7-Flash-Q4_1.gguf"
#MODEL_PATH="/media/iacoppbk/80F42C9BF42C96061/llms/Qwen3-4B-Instruct-2507-Q4_0.gguf"
LOG_FILE="bench_results.md"

BENCH_PARAMS=(
    -m "$MODEL_PATH"       # Model path
    -ngl 99                # Number of GPU layers (all on GPU)
    -t $(nproc)            # Number of CPU threads
    -fa 1                  # Flash attention (1=on, 0=off)
    -ctk q8_0              # KV cache key type (q8_0 quantization)
    -ctv f16               # KV cache value type (f16 precision)
    --main-gpu 0           # Main GPU device ID
    --progress             # Show progress during benchmark
    -r 1                   # Number of repetitions
    -b 2048                # Batch size
    -ub 1536                # Micro-batch size
    #-d 8192               # Context size 
)

# Benchmark tests:
# - Prompt processing (pp): 512, 2048, 8192 tokens
# - Token generation (tg): 128, 512 tokens
# - Depth: 0 (default context)
BENCH_TESTS="-p 512 -n 128"

echo "=== Benchmark ==="
echo "Model: $(basename "$MODEL_PATH")"
echo "Tests: pp512, pp2048, pp8192, tg128, tg512"
echo ""

cd "$(dirname "$0")"
[ ! -f "./build/bin/llama-bench" ] && echo "Error: llama-bench not found" && exit 1

./build/bin/llama-bench "${BENCH_PARAMS[@]}" $BENCH_TESTS "$@" 2>&1 | tee -a "$LOG_FILE"

echo ""
echo "Output saved to: $LOG_FILE"
