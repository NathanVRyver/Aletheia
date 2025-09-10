#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH=${1:-./checkpoints/aletheia-merged}
OUTPUT_DIR=${2:-./cpu_models}

echo "Converting Aletheia model to GGUF format for CPU inference"
echo "Model: $MODEL_PATH"
echo "Output: $OUTPUT_DIR"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Check if model exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model directory $MODEL_PATH does not exist!"
    echo "Run 'make merge' first to create the merged model."
    exit 1
fi

# Check if llama.cpp is available
LLAMA_CPP_DIR="$HOME/llama.cpp"

if [ ! -d "$LLAMA_CPP_DIR" ]; then
    echo "llama.cpp not found. Installing..."
    cd "$HOME"
    git clone https://github.com/ggerganov/llama.cpp.git
    cd llama.cpp
    make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
    cd - > /dev/null
fi

echo "Converting to GGUF..."
cd "$LLAMA_CPP_DIR"

# Convert HF model to GGUF
python3 convert_hf_to_gguf.py \
    --outfile "$PWD/../$(basename $MODEL_PATH).gguf" \
    "$(realpath "$MODEL_PATH")"

BASE_GGUF="$PWD/../$(basename $MODEL_PATH).gguf"

echo "Quantizing model..."

# Create different quantization levels
QUANT_LEVELS=("Q4_K_M" "Q5_K_M" "Q6_K" "Q8_0")

for quant in "${QUANT_LEVELS[@]}"; do
    output_file="$OUTPUT_DIR/aletheia-${quant,,}.gguf"
    echo "Creating $quant quantization: $output_file"
    
    ./llama-quantize "$BASE_GGUF" "$output_file" "$quant"
done

# Move original GGUF to output directory
mv "$BASE_GGUF" "$OUTPUT_DIR/aletheia-f16.gguf"

echo ""
echo "Conversion complete! Available models:"
ls -lh "$OUTPUT_DIR"/*.gguf

echo ""
echo "Model recommendations:"
echo "- aletheia-q4_k_m.gguf: Good balance of quality/speed (recommended)"
echo "- aletheia-q5_k_m.gguf: Better quality, slightly slower"
echo "- aletheia-q6_k.gguf: High quality, slower"
echo "- aletheia-q8_0.gguf: Highest quality, slowest"
echo "- aletheia-f16.gguf: Original precision, largest file"

echo ""
echo "Test with: make cpu-run"