import json, glob
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model
import yaml

def load_jsonl(path_glob):
    rows = []
    for p in glob.glob(path_glob):
        with open(p) as f:
            for line in f: 
                rows.append(json.loads(line))
    return Dataset.from_list([{"text": f"### Instruction:\n{r['prompt']}\n\n### Response:\n{r['response']}"} for r in rows])

def main(cfg):
    tok = AutoTokenizer.from_pretrained(cfg["base_model"], use_fast=True)
    tok.pad_token = tok.eos_token
    ds = load_jsonl(cfg["dataset_path"])
    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"],
        device_map="auto",
        load_in_8bit=cfg["precision"]["load_8bit"]
    )
    lora = LoraConfig(
        r=cfg["lora"]["r"], 
        lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"], 
        target_modules=cfg["lora"]["target_modules"],
        bias="none", 
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora)
    trainer = SFTTrainer(
        model=model, 
        tokenizer=tok, 
        train_dataset=ds,
        args=SFTConfig(
            output_dir=cfg["output_dir"],
            max_seq_length=cfg["max_seq_len"],
            per_device_train_batch_size=cfg["per_device_train_batch_size"],
            gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
            num_train_epochs=cfg["epochs"],
            learning_rate=cfg["lr"],
            lr_scheduler_type=cfg["lr_scheduler"],
            warmup_ratio=cfg["warmup_ratio"],
            weight_decay=cfg["weight_decay"],
            logging_steps=cfg["logging_steps"],
            save_steps=cfg["save_steps"],
            bf16=True
        ),
        dataset_text_field="text",
        packing=cfg["packing"]
    )
    trainer.train()
    trainer.save_model(cfg["output_dir"])

if __name__ == "__main__":
    main(yaml.safe_load(open("training/cfg/sft.yaml")))