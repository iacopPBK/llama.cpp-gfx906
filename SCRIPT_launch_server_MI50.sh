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

# Model path 
MODEL_PATH="/path/..."


# Vision projector path (uncomment for multimodal models)
#MMPROJ_PATH="/path/..."

# Model path .................. -m
# Vision projector ............ --mmproj
# GPU layers (99 = all) ....... -ngl
# Flash attention ............. -fa
# KV cache key type ........... -ctk
# KV cache value type ......... -ctv
# Listen interface ............ --host
# Server port ................. --port
# Context size (tokens) ....... -c
# Jinja templating ............ --jinja


./build/bin/llama-server \
    -m "$MODEL_PATH" \
    -ngl 99 \
    -fa on \
    -ctk q8_0 \
    -ctv f16 \
    --host 0.0.0.0 \
    --port 8080 \
    -c 80000 \
    -b 2048 \
    -ub 1800 \
    --jinja
    # --mmproj "$MMPROJ_PATH"
    
    
    
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

