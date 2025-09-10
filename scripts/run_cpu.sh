#!/usr/bin/env bash
set -euo pipefail

MODEL_FILE=${1:-./cpu_models/aletheia-q4_k_m.gguf}
PROMPT=${2:-"Explain the difference between epoll and kqueue for network servers."}
MAX_TOKENS=${3:-512}

echo "Aletheia CPU Inference"
echo "======================"
echo "Model: $MODEL_FILE"
echo "Prompt: $PROMPT"
echo ""

# Check if model exists
if [ ! -f "$MODEL_FILE" ]; then
    echo "Error: Model file $MODEL_FILE not found!"
    echo "Run 'make cpu-convert' first to create GGUF models."
    exit 1
fi

# Check for llama.cpp
LLAMA_CPP_DIR="$HOME/llama.cpp"
LLAMA_CLI="$LLAMA_CPP_DIR/llama-cli"

if [ ! -f "$LLAMA_CLI" ]; then
    echo "Error: llama-cli not found at $LLAMA_CLI"
    echo "Run 'make cpu-convert' first to install llama.cpp."
    exit 1
fi

# Format prompt for instruction following
FORMATTED_PROMPT="### Instruction:
$PROMPT

### Response:
"

echo "Running inference..."
echo "===================="

# Run llama.cpp with optimized settings for M-series Mac
"$LLAMA_CLI" \
    -m "$MODEL_FILE" \
    -p "$FORMATTED_PROMPT" \
    -n "$MAX_TOKENS" \
    -t $(sysctl -n hw.ncpu) \
    --temp 0.2 \
    --top-p 0.9 \
    --repeat-penalty 1.1 \
    --ctx-size 4096 \
    -ngl 0

echo ""
echo "======================"
echo "Inference complete!"