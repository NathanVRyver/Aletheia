# Aletheia LLM Training Pipeline
# Configuration
SFT_CFG=training/cfg/sft.yaml
DPO_CFG=training/cfg/dpo.yaml
MERGED=checkpoints/aletheia-merged
BASE_MODEL=deepseek-ai/deepseek-coder-7b-instruct-v1.5
PORT=8000

.PHONY: help install sft dpo merge eval custom-eval serve clean setup-data cpu-convert cpu-run

help:
	@echo "Aletheia LLM Training Pipeline"
	@echo "=============================="
	@echo "setup-env     - Create virtual environment and install dependencies"
	@echo "setup-data    - Create sample training data"
	@echo "sft           - Run supervised fine-tuning"
	@echo "dpo           - Run DPO preference optimization"
	@echo "merge         - Merge LoRA adapters into full model"
	@echo "eval          - Run lm-eval-harness benchmarks"
	@echo "custom-eval   - Run custom domain evaluation"
	@echo "serve         - Start vLLM inference server"
	@echo "cpu-convert   - Convert model to GGUF for CPU inference"
	@echo "cpu-run       - Run CPU inference with llama.cpp"
	@echo "clean         - Clean up checkpoints and cache"
	@echo "pipeline      - Run complete training pipeline (sft->dpo->merge->eval)"

setup-env:
	python -m venv .venv
	@echo "Activate environment with: source .venv/bin/activate"
	@echo "Then run: pip install -r requirements.txt"

setup-data:
	python -c "from scripts.create_sample_data import main; main()"

sft:
	@echo "Starting supervised fine-tuning..."
	python training/sft.py

dpo:
	@echo "Starting DPO training..."
	python training/dpo.py

merge:
	@echo "Merging LoRA adapters..."
	python training/lora_merge.py checkpoints/aletheia-dpo $(MERGED)

eval:
	@echo "Running evaluation harness..."
	bash eval/run_harness.sh $(MERGED)

custom-eval:
	@echo "Running custom evaluation..."
	python eval/custom_eval.py $(MERGED)

serve:
	@echo "Starting vLLM server on port $(PORT)..."
	bash serving/start_vllm.sh $(MERGED) $(PORT)

cpu-convert:
	@echo "Converting to GGUF format..."
	bash scripts/convert_to_gguf.sh $(MERGED)

cpu-run:
	@echo "Running CPU inference..."
	bash scripts/run_cpu.sh

pipeline: sft dpo merge eval
	@echo "Complete training pipeline finished!"

clean:
	rm -rf checkpoints/aletheia-*
	rm -rf ~/.cache/huggingface/transformers
	@echo "Cleaned up checkpoints and cache"