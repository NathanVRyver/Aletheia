import json, glob
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DPOTrainer, DPOConfig
from peft import AutoPeftModelForCausalLM
import yaml

def load_pairs(glob_path):
    rows = []
    for p in glob.glob(glob_path):
        with open(p) as f:
            for line in f: 
                rows.append(json.loads(line))
    return Dataset.from_list(rows)

def main(cfg):
    tok = AutoTokenizer.from_pretrained(cfg["base_model"], use_fast=True)
    tok.pad_token = tok.eos_token
    model = AutoPeftModelForCausalLM.from_pretrained(cfg["base_model"], device_map="auto", torch_dtype="bfloat16")
    ds = load_pairs(cfg["dataset_path"])
    trainer = DPOTrainer(
        model=model, 
        tokenizer=tok, 
        beta=cfg["beta"],
        train_dataset=ds, 
        args=DPOConfig(
            output_dir=cfg["output_dir"], 
            num_train_epochs=cfg["epochs"],
            per_device_train_batch_size=cfg["per_device_train_batch_size"],
            gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
            learning_rate=cfg["lr"], 
            logging_steps=20, 
            save_steps=1000, 
            bf16=True
        ),
        formatting_func=lambda ex: (ex["prompt"], ex["chosen"], ex["rejected"])
    )
    trainer.train()
    trainer.save_model(cfg["output_dir"])

if __name__ == "__main__":
    main(yaml.safe_load(open("training/cfg/dpo.yaml")))