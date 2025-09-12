#!/bin/bash
#
# Vector Kernel Inference Profiling - GFX906
# Profile single token generation to capture vector kernels
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROFILE_DIR="${SCRIPT_DIR}/vector_inference_profiling"
MODEL_PATH="/home/iacopo/.cache/llama.cpp/unsloth_Qwen3-Coder-30B-A3B-Instruct-GGUF_Qwen3-Coder-30B-A3B-Instruct-Q4_1.gguf"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_DIR="$PROFILE_DIR/run_inference_$TIMESTAMP"

# ROCm environment - FORCE MI50 ONLY
export HSA_OVERRIDE_GFX_VERSION=9.0.6
export HIP_VISIBLE_DEVICES=0
export ROCR_VISIBLE_DEVICES=0
export GGML_BACKEND_HIP=1

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_step() { echo -e "${BLUE}[STEP]${NC} $1"; }
log_success() { echo -e "${CYAN}[SUCCESS]${NC} $1"; }

cleanup() {
    log_info "Cleaning up..."
    pkill -f llama-server 2>/dev/null || true
    pkill -f rocprof 2>/dev/null || true
    sleep 2
}

trap cleanup EXIT

log_success "=== VECTOR KERNEL INFERENCE PROFILING ==="
log_info "Target: Capture flash_attn_vec kernels during inference"
log_info "Model: $(basename "$MODEL_PATH")"

# Setup directories
mkdir -p "$RUN_DIR/logs"
mkdir -p "$RUN_DIR/traces"
mkdir -p "$RUN_DIR/analysis"

cleanup

log_step "Starting llama-server for inference profiling..."
cd "$SCRIPT_DIR"

# Start server with profiling - configured for inference (small batch)
rocprof --hip-trace --hsa-trace --stats \
    --timestamp on --basenames on \
    -o "$RUN_DIR/vector_inference_profile.csv" \
    -d "$RUN_DIR/traces" \
    ./build/bin/llama-server \
    -m "$MODEL_PATH" \
    -ngl 99 \
    -c 2048 \
    -np 1 \
    -t 8 \
    --port 8081 \
    --host 127.0.0.1 \
    --no-mmap \
    -b 1 \
    --flash-attn on \
    --cache-type-k f16 \
    --cache-type-v f16 \
    --main-gpu 0 \
    --ctx-size 2048 \
    > "$RUN_DIR/logs/server_stdout.log" 2> "$RUN_DIR/logs/server_stderr.log" &

SERVER_PID=$!
log_info "Inference server started with PID: $SERVER_PID"

# Wait for startup
sleep 45

# Test connectivity
for i in {1..5}; do
    if curl -s --max-time 10 "http://127.0.0.1:8081/health" > /dev/null 2>&1; then
        log_success "Server ready for inference!"
        break
    else
        log_info "Waiting for server... ($i/5)"
        sleep 10
    fi
done

# Inference prompt - shorter to focus on token generation
INFERENCE_PROMPT="Write a simple Hello World program in C++:"

log_step "Performing inference to capture vector kernels..."

# Send inference request - multiple small requests to trigger repeated vector kernel usage
for i in {1..3}; do
    log_info "Inference request $i/3"
    timeout 60 curl -s -X POST "http://127.0.0.1:8081/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{
          \"model\": \"qwen3-coder\",
          \"messages\": [{\"role\": \"user\", \"content\": \"$INFERENCE_PROMPT\"}],
          \"max_tokens\": 100,
          \"temperature\": 0.7,
          \"stream\": false
        }" \
        > "$RUN_DIR/logs/inference_response_$i.json" 2>&1 &
    
    # Let it run for a bit then continue
    sleep 20
done

log_info "Letting inference complete and profiler capture data..."
sleep 30

# Stop server
log_step "Stopping inference server..."
kill $SERVER_PID 2>/dev/null || true
sleep 10

# Analyze results
log_step "Analyzing vector kernel profiling results..."

RESULTS_FILE=""
for trace_file in $(find "$RUN_DIR/traces" -name "*results*.txt" 2>/dev/null); do
    if [ -f "$trace_file" ]; then
        RESULTS_FILE="$trace_file"
        break
    fi
done

if [ -z "$RESULTS_FILE" ]; then
    log_info "No profiling results found - checking traces directory"
    ls -la "$RUN_DIR/traces/" || true
    exit 1
fi

log_info "Analyzing: $(basename "$RESULTS_FILE")"

# Search for vector kernels specifically
VECTOR_KERNELS=$(grep -c -i "flash_attn_vec" "$RESULTS_FILE" 2>/dev/null || echo 0)
TILE_KERNELS=$(grep -c -i "flash_attn_tile" "$RESULTS_FILE" 2>/dev/null || echo 0)
TOTAL_KERNELS=$(grep -c "dispatch\[" "$RESULTS_FILE" 2>/dev/null || echo 0)

log_info "Inference Kernel Analysis:"
log_info "  Vector kernels (flash_attn_vec): $VECTOR_KERNELS"
log_info "  Tile kernels (flash_attn_tile): $TILE_KERNELS"  
log_info "  Total kernel dispatches: $TOTAL_KERNELS"

# Extract vector kernels if found
if [ $VECTOR_KERNELS -gt 0 ]; then
    grep -i "flash_attn_vec" "$RESULTS_FILE" > "$RUN_DIR/analysis/vector_kernels.txt" 2>/dev/null || true
    log_success "🎯 VECTOR KERNELS CAPTURED DURING INFERENCE!"
    log_info "Vector kernel details:"
    head -3 "$RUN_DIR/analysis/vector_kernels.txt" | while read line; do
        log_info "  $line"
    done
else
    log_info "⚠️ No vector kernels found - still using tile kernels during inference"
    if [ $TILE_KERNELS -gt 0 ]; then
        grep -i "flash_attn_tile" "$RESULTS_FILE" > "$RUN_DIR/analysis/tile_kernels_inference.txt" 2>/dev/null || true
        log_info "Tile kernels during inference:"
        head -3 "$RUN_DIR/analysis/tile_kernels_inference.txt" | while read line; do
            log_info "  $line"
        done
    fi
fi

# Generate summary
cat > "$RUN_DIR/analysis/inference_summary.md" << EOF
# Vector Kernel Inference Profile Results

**Date:** $(date)
**Model:** Qwen3-Coder-30B Q4_1
**Context:** Single token inference (batch size 1)
**Target:** Capture flash_attn_vec kernels

## Results
- **Vector Kernels:** $VECTOR_KERNELS $([ $VECTOR_KERNELS -gt 0 ] && echo "✅ SUCCESS!" || echo "❌ Not found")
- **Tile Kernels:** $TILE_KERNELS
- **Total Kernels:** $TOTAL_KERNELS

$(if [ $VECTOR_KERNELS -gt 0 ]; then
echo "✅ **SUCCESS: Vector kernels captured during inference!**"
echo ""
echo "This confirms that:"
echo "- Vector kernels ARE used during single token generation"
echo "- Our fattn-vec-f16-gfx906-d128.cuh kernel optimizations are relevant"
echo "- Both tile (prompt) and vector (inference) kernels need optimization"
else
echo "⚠️ **Vector kernels not found during inference**"
echo ""
echo "This suggests either:"
echo "- Batch size still too large (server using tile kernels)"
echo "- Vector kernel selection logic bypassed on GFX906"
echo "- Need to check kernel selection conditions more carefully"
fi)

## Next Steps
$(if [ $VECTOR_KERNELS -gt 0 ]; then
echo "1. Analyze vector kernel performance characteristics"
echo "2. Compare vector vs tile kernel usage patterns"
echo "3. Optimize both kernel types for complete Flash Attention coverage"
else
echo "1. Verify batch size is truly 1 during inference"
echo "2. Check Q->ne[1] values in server logs"
echo "3. Debug kernel selection logic for GFX906"
fi)

EOF

log_success "=== INFERENCE PROFILING COMPLETE ==="
log_info "Results: $RUN_DIR/analysis/"
log_info "Vector kernels found: $VECTOR_KERNELS"
log_info "Analysis: $RUN_DIR/analysis/inference_summary.md"

cleanup
trap - EXIT