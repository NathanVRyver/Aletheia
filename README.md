# Aletheia: End-to-End LLM Training Pipeline

Aletheia is a complete LLM training and serving pipeline focused on systems engineering and backend development expertise. Built on DeepSeek Coder 7B, Aletheia demonstrates a full production workflow: data curation → SFT → DPO → evaluation → serving.

## Project Goals

- **Specialized Model**: Fine-tune DeepSeek Coder 7B for systems/backend engineering tasks
- **Complete Pipeline**: End-to-end workflow from training to production serving
- **Measurable Improvement**: Quantified performance gains over base model
- **Production Ready**: Real-world serving infrastructure with API compatibility

## Architecture

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│    Data     │    │     SFT     │    │     DPO     │    │    Serve    │
│  Curation   │───▶│  Training   │───▶│  Training   │───▶│   & Eval    │
│             │    │             │    │             │    │             │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
       │                   │                   │                   │
   Sample Q&A         LoRA Adapters      Preference        vLLM Server
   Preference         Instruction        Optimization      CPU Inference
   Pairs              Following          Better Alignment   API Endpoints
```

## Quick Start

### 1. Environment Setup
```bash
# Clone and setup
git clone <repo-url>
cd aletheia

# Create virtual environment
make setup-env
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Sample Data
```bash
make setup-data
```

### 3. Run Complete Training Pipeline
```bash
# Full pipeline: SFT → DPO → Merge → Evaluate
make pipeline
```

### 4. Start Serving
```bash
# GPU serving with vLLM
make serve

# CPU serving with llama.cpp
make cpu-convert
make cpu-run
```

## Project Structure

```
aletheia/
├── training/           # Training scripts and configs
│   ├── sft.py         # Supervised fine-tuning
│   ├── dpo.py         # Direct preference optimization
│   ├── lora_merge.py  # LoRA adapter merging
│   └── cfg/           # Training configurations
├── data/              # Training and evaluation data
│   ├── sft/           # SFT training data (prompt/response pairs)
│   ├── dpo/           # DPO preference pairs (chosen/rejected)
│   └── eval/          # Evaluation datasets
├── eval/              # Evaluation scripts
│   ├── run_harness.sh # lm-eval-harness benchmarks
│   └── custom_eval.py # Custom domain evaluation
├── serving/           # Serving infrastructure
│   ├── start_vllm.sh  # vLLM server startup
│   ├── openai_proxy.py # OpenAI-compatible proxy
│   └── test_api.py    # API testing script
├── scripts/           # Utility scripts
│   ├── create_sample_data.py # Sample data generation
│   ├── convert_to_gguf.sh    # GGUF conversion
│   ├── run_cpu.sh            # CPU inference
│   └── cpu_server.py         # CPU API server
└── docs/              # Documentation
```

## Training Process

### Phase 1: Supervised Fine-Tuning (SFT)
- **Base Model**: DeepSeek Coder 7B Instruct
- **Method**: LoRA fine-tuning with 8-bit quantization
- **Data**: Systems engineering Q&A pairs (~6 samples)
- **Focus**: Instruction following and domain knowledge

### Phase 2: Direct Preference Optimization (DPO)
- **Input**: SFT checkpoint
- **Method**: DPO training on preference pairs
- **Data**: Chosen/rejected response pairs (~3 samples)
- **Focus**: Response quality and alignment

### Phase 3: Model Merging
- Merge LoRA adapters into base model
- Create standalone checkpoint for serving

## Evaluation

### Benchmark Evaluation
```bash
make eval
```
Runs lm-eval-harness on standard benchmarks:
- HellaSwag (commonsense reasoning)
- TruthfulQA (truthfulness)
- WinoGrande (reasoning)
- GSM8K (math reasoning)
- MBPP (code generation)

### Custom Evaluation
```bash
make custom-eval
```
Domain-specific evaluation on systems engineering tasks.

## Serving Options

### GPU Serving (vLLM)
```bash
# Start vLLM server
make serve

# Test API
python serving/test_api.py

# Use OpenAI-compatible endpoint
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer aletheia-key-123" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "aletheia",
    "messages": [{"role": "user", "content": "Explain epoll vs kqueue"}],
    "max_tokens": 300
  }'
```

### CPU Serving (llama.cpp)
```bash
# Convert to GGUF and quantize
make cpu-convert

# Run single inference
make cpu-run

# Start HTTP server
python scripts/cpu_server.py
```

## Available Commands

| Command | Description |
|---------|-------------|
| `make help` | Show all available commands |
| `make setup-env` | Create Python virtual environment |
| `make setup-data` | Generate sample training data |
| `make sft` | Run supervised fine-tuning |
| `make dpo` | Run DPO preference optimization |
| `make merge` | Merge LoRA adapters into full model |
| `make eval` | Run benchmark evaluation |
| `make custom-eval` | Run custom domain evaluation |
| `make serve` | Start vLLM GPU server |
| `make cpu-convert` | Convert model to GGUF for CPU |
| `make cpu-run` | Run CPU inference |
| `make pipeline` | Run complete training pipeline |
| `make clean` | Clean checkpoints and cache |

## Configuration

### Training Configuration
Edit `training/cfg/sft.yaml` and `training/cfg/dpo.yaml` to adjust:
- Learning rates and schedules
- Batch sizes and accumulation steps
- LoRA parameters (rank, alpha, dropout)
- Model precision settings

### Data Configuration
- **SFT Data**: Place JSONL files in `data/sft/`
  ```json
  {"prompt": "Your question", "response": "Expected answer"}
  ```
- **DPO Data**: Place JSONL files in `data/dpo/`
  ```json
  {
    "prompt": "Your question",
    "chosen": "Better response",
    "rejected": "Worse response"
  }
  ```

## Customization

### Adding Your Own Data
1. Replace sample data in `data/sft/` and `data/dpo/`
2. Follow the JSONL format shown above
3. Ensure consistent formatting and quality

### Changing Base Model
1. Update `base_model` in `training/cfg/sft.yaml`
2. Adjust LoRA `target_modules` for the new architecture
3. Update evaluation baselines

### Tuning Performance
- **Memory**: Adjust `load_in_8bit`/`load_in_4bit` in configs
- **Speed**: Tune `batch_size` and `gradient_accumulation_steps`
- **Quality**: Increase LoRA rank or training epochs

## Expected Results

With proper training data (1k+ samples), expect:
- **Custom Evaluation**: 5-15% improvement in domain-specific tasks
- **Benchmark Performance**: Maintained or slight improvement
- **Response Quality**: Better structure, technical accuracy, and style

## Troubleshooting

### Common Issues

**CUDA Out of Memory**
- Reduce `per_device_train_batch_size` in configs
- Enable `load_in_4bit` for more aggressive quantization
- Use `gradient_checkpointing: true`

**Training Too Slow**
- Increase `gradient_accumulation_steps`
- Use multiple GPUs with `--multi_gpu`
- Enable `packing: true` for better utilization

**Poor Model Quality**
- Increase training data quantity and quality
- Tune learning rates (try 1e-5 to 5e-4)
- Increase LoRA rank for more capacity

### Hardware Requirements

**Minimum (Training)**
- 16GB GPU RAM (with 4-bit quantization)
- 32GB System RAM
- 100GB free disk space

**Recommended (Training)**
- 40GB+ GPU RAM (A100, H100)
- 64GB+ System RAM
- 500GB SSD storage

**CPU Inference**
- 8GB+ RAM for Q4 quantization
- M-series Mac or modern x86_64 CPU

## Further Reading

- [Training Guide](docs/training.md) - Detailed training instructions
- [Evaluation Guide](docs/evaluation.md) - Comprehensive evaluation methodology
- [Serving Guide](docs/serving.md) - Production deployment patterns
- [Architecture Guide](docs/architecture.md) - Technical deep dive

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add improvement'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- DeepSeek for the base model
- Hugging Face for transformers and training libraries
- vLLM team for efficient serving
- llama.cpp for CPU inference capabilities