from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
import torch
import os
import sys

src = sys.argv[1]  # ./checkpoints/aletheia-dpo
dst = sys.argv[2]  # ./checkpoints/aletheia-merged

tok = AutoTokenizer.from_pretrained(src, use_fast=True)
model = AutoPeftModelForCausalLM.from_pretrained(src, torch_dtype=torch.bfloat16)
model = model.merge_and_unload()  # merges adapters into base
model.save_pretrained(dst, safe_serialization=True)
tok.save_pretrained(dst)