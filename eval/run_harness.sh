#!/usr/bin/env bash
set -euo pipefail
MODEL=${1:-./checkpoints/aletheia-merged}
export HF_HOME=~/.cache/hf
lm_eval \
  --model hf \
  --model_args pretrained="$MODEL",dtype=bfloat16 \
  --tasks hellaswag,truthfulqa_mc2,winogrande,gsm8k,mbpp \
  --batch_size auto \
  --output_path eval/results.json