#!/usr/bin/env bash
set -euo pipefail

MODEL=${1:-./checkpoints/aletheia-merged}
PORT=${2:-8000}
HOST=${3:-0.0.0.0}

echo "Starting vLLM server..."
echo "Model: $MODEL"
echo "Port: $PORT"
echo "Host: $HOST"

# Check if model exists
if [ ! -d "$MODEL" ]; then
    echo "Error: Model directory $MODEL does not exist!"
    echo "Make sure you've run 'make merge' first to create the merged model."
    exit 1
fi

# Start vLLM OpenAI-compatible API server
python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --host "$HOST" \
  --port "$PORT" \
  --served-model-name aletheia \
  --max-model-len 4096 \
  --tensor-parallel-size 1 \
  --dtype bfloat16 \
  --api-key aletheia-key-123 \
  --disable-log-stats