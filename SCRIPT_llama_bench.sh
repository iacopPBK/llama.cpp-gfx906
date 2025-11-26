# Set ROCm environment variables for MI50 ONLY (optimal configuration)
export HSA_OVERRIDE_GFX_VERSION=9.0.6
export HIP_VISIBLE_DEVICES=0           # ONLY MI50 (Device 0)
export CUDA_VISIBLE_DEVICES=0          # Additional CUDA compatibility
export ROCR_VISIBLE_DEVICES=0          # ROCr runtime device selection
export GGML_BACKEND_HIP=1
export HCC_AMDGPU_TARGET=gfx906

# Log output file
LOG_FILE="bench_results.md"

# Path to your model file - update this to your actual model path
MODEL_PATH="/home/iacopo/Downloads/Qwen3-4B-Instruct-2507-Q4_0.gguf"

# Default benchmark parameters (matching server configuration)
BENCH_PARAMS=(
    -m "$MODEL_PATH"
    -ngl 99                    # Offload all layers to GPU
    #-b 4096                    # Batch size (matches server)
    #-ub 4096
    -t $(nproc)                # Use all CPU threads
    -fa 1                      # Enable flash attention (GFX906 optimized)
    -ctk q8_0                  # q8_0 quantized K cache (matches server)
    -ctv f16                  # q8_0 quantized V cache (matches server)
    --main-gpu 0               # Force MI50 as main GPU
    --progress                 # Show progress indicators
    -r 1                       # Only 1 run after warmup (not 5 default)
    -d 2048
)

# Benchmark configurations
QUICK_TEST="-p 1024 -n 128"
STANDARD_TEST="-p 512 -n 128"
PROMPT_FOCUS="-p 512,1024,2048,4096,8192 -n 64"
GENERATION_FOCUS="-p 512 -n 128,256,512,1024"
EXTENSIVE_TEST="-p 512,1024,2048,4096,8192 -n 128,256,512"

# Function to display usage
usage() {
    echo "Usage: $0 [test_type] [additional_llama-bench_args...]"
    echo ""
    echo "Test types:"
    echo "  quick       - Quick test (512 prompt, 128 generation)"
    echo "  standard    - Standard test (multiple prompt sizes, 2 gen sizes) [DEFAULT]"
    echo "  prompt      - Focus on prompt processing (up to 8K prompts)"
    echo "  generation  - Focus on text generation (multiple lengths)"
    echo "  extensive   - Extensive testing (all combinations)"
    echo "  custom      - Use your own parameters (provide as additional args)"
    echo ""
    echo "Examples:"
    echo "  $0                          # Run standard benchmark"
    echo "  $0 quick                    # Run quick benchmark"
    echo "  $0 prompt                   # Test prompt processing"
    echo "  $0 custom -p 1024 -n 256    # Custom benchmark"
    echo ""
    echo "Model path: $MODEL_PATH"

}

# Parse command line arguments
TEST_TYPE="${1:-standard}"
shift  # Remove first argument, rest will be passed to llama-bench

case "$TEST_TYPE" in
    "help"|"-h"|"--help")
        usage
        exit 0
        ;;
    "quick")
        TEST_PARAMS="$QUICK_TEST"
        echo "=== Running Quick Benchmark ==="
        ;;
    "standard")
        TEST_PARAMS="$STANDARD_TEST"
        echo "=== Running Standard Benchmark ==="
        ;;
    "prompt")
        TEST_PARAMS="$PROMPT_FOCUS"
        echo "=== Running Prompt Processing Focused Benchmark ==="
        ;;
    "generation")
        TEST_PARAMS="$GENERATION_FOCUS"
        echo "=== Running Text Generation Focused Benchmark ==="
        ;;
    "extensive")
        TEST_PARAMS="$EXTENSIVE_TEST"
        echo "=== Running Extensive Benchmark (this will take a while) ==="
        ;;
    "custom")
        TEST_PARAMS=""
        echo "=== Running Custom Benchmark ==="
        echo "Custom parameters: $@"
        ;;
    *)
        echo "Unknown test type: $TEST_TYPE"
        usage
        exit 1
        ;;
esac

# Display system info
echo "Model: $(basename "$MODEL_PATH")"
echo ""

# Display GPU info
echo "=== ROCm GPU Information ==="
rocm-smi --showproductname --showtemp --showmeminfo --showuse --showpower
echo ""

# Change to script directory
cd "$(dirname "$0")"

# Check if llama-bench exists
if [ ! -f "./build/bin/llama-bench" ]; then
    echo "Error: llama-bench not found. Please compile the project first:"
    echo "  ./SCRIPT_compile_MI50.sh"
    exit 1
fi

# Run the benchmark
echo "Command: ./build/bin/llama-bench ${BENCH_PARAMS[*]} $TEST_PARAMS $@"
echo ""
echo "Output will be saved to: $LOG_FILE"
echo ""

# Run benchmark and capture ALL output to both terminal and log file
./build/bin/llama-bench "${BENCH_PARAMS[@]}" $TEST_PARAMS "$@" 2>&1 | tee -a "$LOG_FILE"

BENCH_EXIT_CODE=${PIPESTATUS[0]}

# Close the code block and add summary
{
    echo "\`\`\`"
    echo ""
    echo "## Flash Attention Kernel Launches"
    echo "\`\`\`"
    grep -i "flash_attn_tile_q8\|ncols=64\|v_dot4" "$LOG_FILE" | head -20 || echo "No flash_attn_tile_q8 kernel traces found"
    echo "\`\`\`"
    echo ""
    echo "## Benchmark Status"
    if [ $BENCH_EXIT_CODE -eq 0 ]; then
        echo "**SUCCESS** - Benchmark completed successfully"
    else
        echo "**FAILED** - Exit code: $BENCH_EXIT_CODE"
    fi
} >> "$LOG_FILE"

# Show flash attention kernel launches
echo ""
echo "=== Flash Attention Kernel Launches ==="
grep -i "flash_attn_tile_q8\|ncols=64\|v_dot4" "$LOG_FILE" | head -20 || echo "No flash_attn_tile_q8 kernel traces found"

echo ""
echo "=== Benchmark Complete ==="
echo "Full output saved to: $LOG_FILE"

exit $BENCH_EXIT_CODE
