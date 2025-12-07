#!/bin/bash
cat << 'EOF'

   ██╗     ██╗      █████╗ ███╗   ███╗ █████╗    ██████╗██████╗ ██████╗
   ██║     ██║     ██╔══██╗████╗ ████║██╔══██╗  ██╔════╝██╔══██╗██╔══██╗
   ██║     ██║     ███████║██╔████╔██║███████║  ██║     ██████╔╝██████╔╝
   ██║     ██║     ██╔══██║██║╚██╔╝██║██╔══██║  ██║     ██╔═══╝ ██╔═══╝      
   ███████╗███████╗██║  ██║██║ ╚═╝ ██║██║  ██║  ╚██████╗██║     ██║        
           ██████╗ ███████╗██╗  ██╗ █████╗  ██████╗  ██████╗                 
          ██╔════╝ ██╔════╝╚██╗██╔╝██╔══██╗██╔═████╗██╔════╝                
          ██║  ███╗█████╗   ╚███╔╝ ╚██████║██║██╔██║███████╗                
          ██║   ██║██╔══╝   ██╔██╗  ╚═══██║████╔╝██║██╔═══██╗                
          ╚██████╔╝██║     ██╔╝ ██╗ █████╔╝╚██████╔╝╚██████╔╝                


EOF

export HSA_OVERRIDE_GFX_VERSION=9.0.6
export HIP_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=0
export ROCR_VISIBLE_DEVICES=0
export GGML_BACKEND_HIP=1
export HCC_AMDGPU_TARGET=gfx906
MODEL_PATH="/media/iacoppbk/80F42C9BF42C9606/llms/openai_gpt-oss-20b-MXFP4.gguf"
#MODEL_PATH="/media/iacoppbk/80F42C9BF42C9606/llms/Qwen3-VL-30B-A3B-Thinking-Q4_1.gguf"
#MODEL_PATH="/media/iacoppbk/80F42C9BF42C9606/llms/Qwen3-4B-Instruct-2507-Q4_1.gguf"
#MODEL_PATH="/media/iacoppbk/80F42C9BF42C9606/llms/Qwen3-4B-Instruct-2507-Q8_0.gguf"
#MODEL_PATH="/media/iacoppbk/80F42C9BF42C9606/llms/openai_gpt-oss-20b-Q4_1.gguf"
LOG_FILE="bench_results.md"

BENCH_PARAMS=(
    -m "$MODEL_PATH"
    -ngl 99
    -t $(nproc)
    -fa 1
    -ctk q8_0
    -ctv f16
    --main-gpu 0
    --progress
    -r 1
    #-d 8192
)

QUICK_TEST="-p 1024 -n 128"
STANDARD_TEST="-p 512 -n 128"
PROMPT_FOCUS="-p 512,1024,2048,4096,8192 -n 64"
GENERATION_FOCUS="-p 512 -n 128,256,512,1024"
EXTENSIVE_TEST="-p 512,1024,2048,4096,8192 -n 128,256,512"

usage() {
    echo "Usage: $0 [quick|standard|prompt|generation|extensive|custom] [args...]"
    echo "Model: $MODEL_PATH"
}

TEST_TYPE="${1:-standard}"
shift 2>/dev/null

case "$TEST_TYPE" in
    "help"|"-h"|"--help") usage; exit 0 ;;
    "quick") TEST_PARAMS="$QUICK_TEST" ;;
    "standard") TEST_PARAMS="$STANDARD_TEST" ;;
    "prompt") TEST_PARAMS="$PROMPT_FOCUS" ;;
    "generation") TEST_PARAMS="$GENERATION_FOCUS" ;;
    "extensive") TEST_PARAMS="$EXTENSIVE_TEST" ;;
    "custom") TEST_PARAMS="" ;;
    *) echo "Unknown: $TEST_TYPE"; usage; exit 1 ;;
esac

echo "=== $TEST_TYPE benchmark ==="
echo "Model: $(basename "$MODEL_PATH")"

cd "$(dirname "$0")"
[ ! -f "./build/bin/llama-bench" ] && echo "Error: llama-bench not found" && exit 1

./build/bin/llama-bench "${BENCH_PARAMS[@]}" $TEST_PARAMS "$@" 2>&1 | tee -a "$LOG_FILE"

echo "Output saved to: $LOG_FILE"
