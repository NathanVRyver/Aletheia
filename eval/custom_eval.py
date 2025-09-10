import json
import glob
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm

def load_eval_data(path_glob):
    rows = []
    for p in glob.glob(path_glob):
        with open(p) as f:
            for line in f:
                rows.append(json.loads(line))
    return rows

def generate_response(model, tokenizer, prompt, max_tokens=512):
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.2,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

def main():
    import sys
    tuned_model_path = sys.argv[1] if len(sys.argv) > 1 else "./checkpoints/aletheia-merged"
    base_model_path = "deepseek-ai/deepseek-coder-7b-instruct-v1.5"
    
    eval_data = load_eval_data("./data/eval/*.jsonl")
    
    print(f"Loading base model: {base_model_path}")
    base_tok = AutoTokenizer.from_pretrained(base_model_path)
    base_tok.pad_token = base_tok.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path, 
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    print(f"Loading tuned model: {tuned_model_path}")
    tuned_tok = AutoTokenizer.from_pretrained(tuned_model_path)
    tuned_tok.pad_token = tuned_tok.eos_token
    tuned_model = AutoModelForCausalLM.from_pretrained(
        tuned_model_path, 
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    results = []
    for item in tqdm(eval_data[:100]):  # limit to 100 for quick eval
        prompt = item["prompt"]
        base_response = generate_response(base_model, base_tok, prompt)
        tuned_response = generate_response(tuned_model, tuned_tok, prompt)
        
        results.append({
            "prompt": prompt,
            "base_response": base_response,
            "tuned_response": tuned_response,
            "expected": item.get("expected", "")
        })
    
    with open("eval/custom_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Evaluated {len(results)} samples. Results saved to eval/custom_results.json")

if __name__ == "__main__":
    main()