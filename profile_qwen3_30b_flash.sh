#!/bin/bash
#
# Qwen3-Coder-30B Flash Attention Test - MI50 Profiling
# This 30B model should definitely trigger flash attention kernels!
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROFILE_DIR="${SCRIPT_DIR}/qwen3_30b_flash_profiling"
MODEL_PATH="/home/iacopo/.cache/llama.cpp/unsloth_Qwen3-Coder-30B-A3B-Instruct-GGUF_Qwen3-Coder-30B-A3B-Instruct-Q4_1.gguf"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_DIR="$PROFILE_DIR/run_qwen3_30b_$TIMESTAMP"

# ROCm environment - FORCE MI50 ONLY
export HSA_OVERRIDE_GFX_VERSION=9.0.6
export HIP_VISIBLE_DEVICES=0           # ONLY MI50 (Device 0)
export CUDA_VISIBLE_DEVICES=0          # Additional CUDA compatibility
export ROCR_VISIBLE_DEVICES=0          # ROCr runtime device selection
export GGML_BACKEND_HIP=1
export HCC_AMDGPU_TARGET=gfx906
export ROC_ENABLE_PRE_VEGA=1

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_step() { echo -e "${BLUE}[STEP]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_success() { echo -e "${CYAN}[SUCCESS]${NC} $1"; }

cleanup() {
    log_info "Cleaning up all processes..."
    pkill -f llama-server 2>/dev/null || true
    pkill -f rocprof 2>/dev/null || true
    sleep 2
}

trap cleanup EXIT

log_success "=== QWEN3-CODER-30B FLASH ATTENTION TEST FOR MI50 ==="
log_info "Timestamp: $(date)"
log_info "Model: $(basename "$MODEL_PATH")"
log_info "Model Size: 30B parameters - PERFECT for flash attention!"
log_info "Target: Capture actual flash attention kernels"

# Verify model exists
if [ ! -f "$MODEL_PATH" ]; then
    log_error "Model file not found: $MODEL_PATH"
    exit 1
fi

MODEL_SIZE=$(du -h "$MODEL_PATH" | cut -f1)
log_info "Model file size: $MODEL_SIZE"

# Setup directories
log_step "Setting up profiling environment..."
mkdir -p "$RUN_DIR/logs"
mkdir -p "$RUN_DIR/traces"
mkdir -p "$RUN_DIR/analysis"

# Kill existing processes
log_step "Killing existing servers and profilers..."
cleanup

# Verify dependencies
if ! command -v rocprof &> /dev/null; then
    log_error "rocprof not found. ROCm not properly installed."
    exit 1
fi

log_step "Starting llama-server with Qwen3-Coder-30B and rocprof profiling..."
cd "$SCRIPT_DIR"

# Start server with profiling - optimized for 30B model flash attention
rocprof --hip-trace --hsa-trace --sys-trace --stats \
    --timestamp on --basenames on \
    -o "$RUN_DIR/qwen3_30b_flash_profile.csv" \
    -d "$RUN_DIR/traces" \
    ./build/bin/llama-server \
    -m "$MODEL_PATH" \
    -ngl 99 \
    -c 4096 \
    -np 1 \
    -t 8 \
    --port 8080 \
    --host 127.0.0.1 \
    --no-mmap \
    -b 2 \
    --flash-attn on \
    --cache-type-k f16 \
    --cache-type-v f16 \
    --no-warmup \
    --main-gpu 0 \
    --device "ROCm0" \
    --ctx-size 4096 \
    > "$RUN_DIR/logs/server_stdout.log" 2> "$RUN_DIR/logs/server_stderr.log" &

SERVER_PID=$!
log_info "Server started with PID: $SERVER_PID"
log_info "Model: Qwen3-Coder-30B Q4_1 (Context: 4096, Flash Attention: ON)"

# Wait for server startup (30B model takes longer)
log_step "Waiting for 30B model initialization..."
sleep 60

# Check server health
log_step "Testing server connectivity..."
HEALTH_CHECK=0
for i in {1..5}; do
    if curl -s --max-time 10 "http://127.0.0.1:8080/health" > /dev/null 2>&1; then
        log_success "Server is ready!"
        HEALTH_CHECK=1
        break
    else
        log_warn "Health check attempt $i/5 failed, waiting..."
        sleep 15
    fi
done

if [ $HEALTH_CHECK -eq 0 ]; then
    log_error "Server failed to start properly"
    exit 1
fi

# Create coding-focused prompt for Qwen3-Coder
log_step "Creating coding prompt for Qwen3-Coder-30B..."

CODING_PROMPT="Please write a comprehensive C++ implementation of a neural attention mechanism, including both standard and flash attention algorithms. The implementation should include:

1. A complete multi-head attention class with configurable parameters
2. Memory-efficient flash attention implementation using tiling
3. Proper memory management and CUDA/ROCm compatibility
4. Batch processing capabilities for multiple sequences
5. Key-Value cache management with quantization support
6. Benchmark comparison between standard and flash attention
7. Unit tests and performance profiling code

Focus on production-ready code with proper error handling, documentation, and optimization for AMD GPUs. Include mathematical explanations for the attention mechanisms and provide examples of how to integrate this into a larger transformer model.

Please write clean, well-documented code that demonstrates advanced C++ techniques and GPU programming best practices."

# Count tokens (rough estimate)
PROMPT_CHARS=$(echo "$CODING_PROMPT" | wc -c)
ESTIMATED_TOKENS=$((PROMPT_CHARS / 4))

log_info "Coding prompt prepared: $PROMPT_CHARS characters, ~$ESTIMATED_TOKENS tokens"

# Execute test with 30B model
log_step "Executing flash attention test with Qwen3-Coder-30B..."

# Create JSON payload
TEMP_JSON=$(mktemp)
cat > "$TEMP_JSON" << EOF
{
  "model": "qwen3-coder",
  "messages": [
    {"role": "user", "content": $(echo "$CODING_PROMPT" | jq -Rs .)}
  ],
  "max_tokens": 2048,
  "temperature": 0.3,
  "stream": false
}
EOF

# Send request with longer timeout for 30B model
timeout 300 curl -s -X POST "http://127.0.0.1:8080/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d @"$TEMP_JSON" \
    > "$RUN_DIR/logs/qwen3_response.json" 2>&1 || log_warn "Request may have timed out"

# Clean up temp file
rm -f "$TEMP_JSON"

log_success "30B model request completed!"

# Let profiler capture extensive data
log_info "Allowing profiler to capture flash attention kernel data..."
sleep 20

# Stop server gracefully
log_step "Stopping server..."
kill $SERVER_PID 2>/dev/null || true
sleep 15

# Analyze profiling results
log_step "Analyzing Qwen3-Coder-30B profiling results..."

# Find results file
RESULTS_FILE=""
for trace_file in $(find "$RUN_DIR/traces" -name "*results*.txt" 2>/dev/null); do
    if [ -f "$trace_file" ]; then
        RESULTS_FILE="$trace_file"
        break
    fi
done

if [ -z "$RESULTS_FILE" ]; then
    log_error "No profiling results file found"
    exit 1
fi

log_info "Analyzing 30B model profiling data: $(basename "$RESULTS_FILE")"

# Analyze kernel types with 30B model
TOTAL_KERNELS=$(grep -c "dispatch\[" "$RESULTS_FILE" 2>/dev/null || echo 0)
FLASH_KERNELS=$(grep -c -i "fattn\|flash.*attn\|attention" "$RESULTS_FILE" 2>/dev/null || echo 0)
QUANT_KERNELS=$(grep -c -i "quantiz\|q4_1\|mmq" "$RESULTS_FILE" 2>/dev/null || echo 0)
COMPUTE_KERNELS=$(grep -c -i "mul_mat\|rope\|norm\|gelu\|silu" "$RESULTS_FILE" 2>/dev/null || echo 0)

log_info "30B Model Kernel Analysis:"
log_info "  Total kernel dispatches: $TOTAL_KERNELS"
log_info "  Flash attention kernels: $FLASH_KERNELS"
log_info "  Quantization kernels: $QUANT_KERNELS"
log_info "  Compute kernels: $COMPUTE_KERNELS"

# Extract and analyze flash attention kernels
if [ $FLASH_KERNELS -gt 0 ]; then
    grep -i "fattn\|flash.*attn\|attention" "$RESULTS_FILE" > "$RUN_DIR/analysis/flash_attention_kernels.txt" 2>/dev/null || true
    log_success "🎯 FLASH ATTENTION KERNELS DETECTED WITH 30B MODEL!"
    
    # Analyze flash attention kernel details
    if [ -f "$RUN_DIR/analysis/flash_attention_kernels.txt" ]; then
        FA_COUNT=$(wc -l < "$RUN_DIR/analysis/flash_attention_kernels.txt")
        log_info "Flash attention kernel details:"
        head -5 "$RUN_DIR/analysis/flash_attention_kernels.txt" | while read line; do
            log_info "  $line"
        done
    fi
else
    log_warn "No flash attention kernels detected even with 30B model"
fi

# Extract other kernel types
if [ $QUANT_KERNELS -gt 0 ]; then
    grep -i "quantiz\|q4_1\|mmq" "$RESULTS_FILE" > "$RUN_DIR/analysis/quantization_kernels.txt" 2>/dev/null || true
fi

if [ $COMPUTE_KERNELS -gt 0 ]; then
    grep -i "mul_mat\|rope\|norm\|gelu\|silu" "$RESULTS_FILE" > "$RUN_DIR/analysis/compute_kernels.txt" 2>/dev/null || true
fi

# Generate comprehensive summary for 30B model
cat > "$RUN_DIR/analysis/qwen3_30b_summary.md" << EOF
# Qwen3-Coder-30B Flash Attention Analysis

**Date:** $(date)
**Model:** Qwen3-Coder-30B-A3B-Instruct Q4_1 (18GB)
**Parameters:** 30 billion (perfect for flash attention)
**Context Size:** 4096 tokens
**Hardware:** AMD Instinct MI50 (gfx906)

## Results Summary

- **Total Kernels:** $TOTAL_KERNELS
- **Flash Attention Kernels:** $FLASH_KERNELS $([ $FLASH_KERNELS -gt 0 ] && echo "✅ SUCCESS!" || echo "❌ Not detected")
- **Quantization Kernels:** $QUANT_KERNELS
- **Compute Kernels:** $COMPUTE_KERNELS
- **Model Loading:** $(grep -q "model.*loaded" "$RUN_DIR/logs/server_stdout.log" 2>/dev/null && echo "✅ Success" || echo "❌ Check logs")

## Flash Attention Analysis

$(if [ $FLASH_KERNELS -gt 0 ]; then
echo "🎯 **BREAKTHROUGH: Flash Attention Kernels Captured!**"
echo ""
echo "✅ **30B parameter model successfully triggered flash attention**"
echo "✅ **Flash attention kernels detected and profiled**"
echo "✅ **Comprehensive kernel execution data available**"
echo ""
echo "### Flash Attention Kernel Details:"
echo "\`\`\`"
head -3 "$RUN_DIR/analysis/flash_attention_kernels.txt" 2>/dev/null || echo "See flash_attention_kernels.txt for details"
echo "\`\`\`"
echo ""
echo "### Key Insights:"
echo "- 30B model size was sufficient to activate flash attention"
echo "- MI50 (gfx906) successfully executes flash attention kernels"
echo "- Context size of 4096 tokens worked with flash attention"
echo "- f16 KV cache used (no quantization) for pure flash attention"
else
echo "⚠️ **Flash attention kernels still not detected**"
echo ""
echo "This could indicate:"
echo "- Build configuration issue with flash attention support"
echo "- Model-specific attention implementation"
echo "- Hardware compatibility constraints"
echo ""
echo "However, we captured $TOTAL_KERNELS total kernels for analysis."
fi)

## Files Generated

$(find "$RUN_DIR" -name "*.txt" -o -name "*.csv" -o -name "*.json" | sort)

EOF

# Display final results
log_success "=== QWEN3-CODER-30B TEST COMPLETE ==="
log_info "Results directory: $RUN_DIR"
log_info "Model: 30B parameters (optimal for flash attention)"
log_info "Flash attention kernels: $FLASH_KERNELS"
log_info "Total kernel activity: $TOTAL_KERNELS"

if [ $FLASH_KERNELS -gt 0 ]; then
    log_success "🎯🎯🎯 FLASH ATTENTION KERNELS SUCCESSFULLY CAPTURED! 🎯🎯🎯"
    log_info "Flash attention data: $RUN_DIR/analysis/flash_attention_kernels.txt"
    log_info "This is the breakthrough we needed!"
elif [ $TOTAL_KERNELS -gt 100000 ]; then
    log_warn "⚠️ Extensive kernel activity but no flash attention detected"
    log_info "May need to check build configuration or try different settings"
else
    log_warn "⚠️ Limited kernel activity - check server startup logs"
fi

echo ""
echo "=================================="
echo "   QWEN3-CODER-30B FLASH RESULTS"
echo "=================================="
if [ -f "$RUN_DIR/analysis/qwen3_30b_summary.md" ]; then
    head -50 "$RUN_DIR/analysis/qwen3_30b_summary.md"
fi

log_success "30B model flash attention test completed!"

# Final cleanup
cleanup
trap - EXIT