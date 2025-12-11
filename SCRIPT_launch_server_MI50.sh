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


./build/bin/llama-server \
    -m "$MODEL_PATH" \      # Model path
    -ngl 99 \               # Number of GPU layers (99 = all layers on GPU)
    -fa on \                # Flash attention enabled
    -ctk q8_0 \             # KV cache key type (q8_0 quantization)
    -ctv f16 \              # KV cache value type (f16 precision)
    --host 0.0.0.0 \        # Listen on all interfaces
    --port 8080 \           # Server port
    -c 80000  \             # Context size (tokens)
    --jinja                 # Enable Jinja templating
# ============================================================================
# DUAL MI50 SERVER CONFIG (@fuutott server config)
# ============================================================================
# HIP_VISIBLE_DEVICES=0,1 llama-server \
#   -hf lmstudio-community/gpt-oss-120b-GGUF \  # HuggingFace model repo
#   --gpu-layers 999 \      # GPU layers (999 = all on GPU)
#   -t 7 \                  # Number of CPU threads
#   --ctx-size 72000 \      # Context size (tokens)
#   --jinja \               # Enable Jinja templating
#   --host 0.0.0.0 \        # Listen on all interfaces
#   --flash-attn on \       # Flash attention enabled
#   --no-mmap \             # Disable memory mapping (required for multi-GPU)
#   --cache-type-k q8_0 \   # KV cache key type (q8_0 quantization)
#   --cache-type-v q8_0 \   # KV cache value type (q8_0 quantization)
#   --chat-template-kwargs '{"reasoning_effort": "medium"}'  # Chat template args
# ============================================================================

