#!/bin/bash
#
# Launch llama.cpp server with AMD MI50 ROCm support
# Built for gfx906 architecture
# Uses ROCm Nightly Runtime for optimized performance
#

# Load ROCm nightly environment for runtime
if [[ -f "$HOME/rocm-nightly-env.sh" ]]; then
    source "$HOME/rocm-nightly-env.sh"
    echo "=== Using ROCm Nightly Runtime ==="
    echo "  Path: $ROCM_PATH"
    echo "  HIP Version: $(hipcc --version 2>/dev/null | grep "HIP version" | head -1 || echo "Unknown")"
    echo ""

    # Override LD_LIBRARY_PATH to use nightly libraries at runtime
    export LD_LIBRARY_PATH="$ROCM_PATH/lib:$LD_LIBRARY_PATH"
else
    echo "WARNING: ROCm nightly not found at ~/rocm-nightly-env.sh"
    echo "Falling back to system ROCm..."
    echo ""
fi

# Set ROCm environment variables for MI50 ONLY (optimal configuration)
export HSA_OVERRIDE_GFX_VERSION=9.0.6
export HIP_VISIBLE_DEVICES=0           # ONLY MI50 (Device 0)
export CUDA_VISIBLE_DEVICES=0          # Additional CUDA compatibility
export ROCR_VISIBLE_DEVICES=0          # ROCr runtime device selection
export GGML_BACKEND_HIP=1
export HCC_AMDGPU_TARGET=gfx906

# Path to your model file - update this to your actual model path
MODEL_PATH="/home/iacopo/Downloads/Qwen3-VL-30B-A3B-Instruct-Q4_1.gguf"
#MODEL_PATH="/home/iacopo/Downloads/Qwen3-4B-Instruct-2507-Q4_0.gguf"

# Path to multimodal projector (required for vision models like Qwen3-VL)
MMPROJ_PATH="/home/iacopo/Downloads/mmproj-F16.gguf"


PARAMS=(
    -m "$MODEL_PATH"
    --mmproj "$MMPROJ_PATH"    # Vision projector for multimodal support
    --jinja                    # Use Jinja chat template (required for Qwen3-VL)
    -ngl 99                    # Offload all layers to GPU
    -c 80000                     # Context size (reduced for M-RoPE compatibility)
    -np 1                      # Parallel requests
    -t $(nproc)                # Use all CPU threads
    --port 8090                # Server port
    --host 0.0.0.0            # Listen on all interfaces
    #--keep 1024
    #--mlock                    # Lock model in memory
    #--no-mmap                  # Don't use memory mapping
    -b 2048                     # Batch size (increased for image tokens)
    #--cont-batching            # Enable continuous batching
    --flash-attn on            # Disable flash attention (M-RoPE compatibility issue)
    --cache-type-k q8_0         # Use F16 cache for M-RoPE (quantized cache causes issues)
    --cache-type-v f16         # Use F16 cache for M-RoPE (quantized cache causes issues)
    --main-gpu 0               # Force MI50 as main GPU
    --device "ROCm0"           # Explicit ROCm device
    # --no-warmup                # Skip warmup for consistent profiling

    --top-p 0.8 \
    --top-k 20 \
    --temp 0.7 \
    --min-p 0.0 \

    --presence-penalty 1.5 \




)

# Check if model file exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model file not found at: $MODEL_PATH"
    echo "Usage: $0 [model_path] [additional_args...]"
    echo ""
    echo "Example: $0 ./models/llama-2-7b-chat.q4_0.gguf --ctx-size 8192"
    exit 1
fi

# Check if mmproj file exists (required for vision models)
if [ ! -f "$MMPROJ_PATH" ]; then
    echo "Error: Multimodal projector file not found at: $MMPROJ_PATH"
    echo "For vision models like Qwen3-VL, you need both:"
    echo "  1. Model GGUF file: $MODEL_PATH"
    echo "  2. mmproj GGUF file: $MMPROJ_PATH"
    echo ""
    echo "Download the mmproj file from Hugging Face or convert with:"
    echo "  python3 convert_hf_to_gguf.py /path/to/model --mmproj"
    exit 1
fi

# Display GPU info
echo "=== ROCm GPU Information ==="
rocm-smi --showproductname --showtemp --showmeminfo --showuse --showpower
echo ""

# Launch llama.cpp server
echo "=== Launching llama.cpp server with MI50 optimization ==="
echo "Model: $MODEL_PATH"
echo "Vision Projector (mmproj): $MMPROJ_PATH"
echo "GPU: MI50 (gfx906)"
echo "Server will be available at: http://localhost:8090"
echo "Multimodal: ENABLED (images supported via web UI)"
echo "Parameters: ${PARAMS[*]} ${@:2}"
echo ""

cd "$(dirname "$0")"
./build/bin/llama-server "${PARAMS[@]}" "${@:2}"
