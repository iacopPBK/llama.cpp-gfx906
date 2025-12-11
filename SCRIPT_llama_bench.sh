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

MODEL_PATH="/mnt/data/openai_gpt-oss-20b-MXFP4.gguf"
#MODEL_PATH="/mnt/data/Qwen3-VL-30B-A3B-Thinking-Q4_1.gguf"
#MODEL_PATH="/mnt/data/Qwen3-4B-Instruct-2507-Q4_0.gguf"
#MODEL_PATH="/mnt/data/Qwen3-4B-Instruct-2507-Q4_1.gguf"
#MODEL_PATH="/mnt/data/Qwen3-4B-Instruct-2507-Q8_0.gguf"
#MODEL_PATH="/mnt/data/Qwen3-Next-80B-A3B-Instruct-Q2_K.gguf"


LOG_FILE="bench_results.md"

BENCH_PARAMS=(
    -m "$MODEL_PATH"       # Model path
    -ngl 99                # Number of GPU layers (99 = all layers on GPU)
    -t $(nproc)            # Number of CPU threads
    -fa 1                  # Flash attention (1=on, 0=off)
    -ctk q8_0              # KV cache key type (q8_0 quantization)
    -ctv f16               # KV cache value type (f16 precision)
    --main-gpu 0           # Main GPU device ID
    --progress             # Show progress during benchmark
    -r 1                   # Number of repetitions
    #-d 8192               # Context size 
)

STANDARD_TEST="-p 512 -n 128"  # -p: prompt tokens, -n: generation tokens

# ============================================================================
# DUAL MI50 CONFIG (@fuutott)
# ============================================================================
# HIP_VISIBLE_DEVICES=0,1 llama-bench \
#   -m ~/.cache/llama.cpp/gpt-oss-120b-MXFP4.gguf \  # Model path
#   -p 512 \              # Prompt tokens to process
#   -n 128 \              # Number of tokens to generate
#   -ngl 99 \             # GPU layers (99 = all on GPU)
#   -mmp 0 \              # Memory map (0=disabled for multi-GPU)
#   -fa 1 \               # Flash attention enabled
#   -o md \               # Output format (markdown)
#   -r 3 \                # Number of repetitions
#   -d 0,8192 \           # Context sizes to test (0=default, 8192)
#   --main-gpu 0          # Primary GPU device ID
# ============================================================================

usage() {
    echo "Model: $MODEL_PATH"
}


echo "=== benchmark ==="
echo "Model: $(basename "$MODEL_PATH")"

cd "$(dirname "$0")"
[ ! -f "./build/bin/llama-bench" ] && echo "Error: llama-bench not found" && exit 1

./build/bin/llama-bench "${BENCH_PARAMS[@]}" $STANDARD_TEST "$@" 2>&1 | tee -a "$LOG_FILE"

echo "Output saved to: $LOG_FILE"
