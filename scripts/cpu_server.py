#!/usr/bin/env python3
"""
Simple HTTP server for CPU inference with llama.cpp
"""
import subprocess
import json
import os
from flask import Flask, request, jsonify
from datetime import datetime
import time

app = Flask(__name__)

# Configuration
MODELS_DIR = "./cpu_models"
DEFAULT_MODEL = "aletheia-q4_k_m.gguf"
LLAMA_CPP_DIR = os.path.expanduser("~/llama.cpp")
LLAMA_CLI = os.path.join(LLAMA_CPP_DIR, "llama-cli")

def run_inference(prompt, model_file=None, max_tokens=512, temperature=0.2):
    """Run inference using llama-cli"""
    if model_file is None:
        model_file = os.path.join(MODELS_DIR, DEFAULT_MODEL)
    
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Model file not found: {model_file}")
    
    if not os.path.exists(LLAMA_CLI):
        raise FileNotFoundError(f"llama-cli not found: {LLAMA_CLI}")
    
    # Format prompt
    formatted_prompt = f"### Instruction:\n{prompt}\n\n### Response:\n"
    
    # Build command
    cmd = [
        LLAMA_CLI,
        "-m", model_file,
        "-p", formatted_prompt,
        "-n", str(max_tokens),
        "-t", str(os.cpu_count()),
        "--temp", str(temperature),
        "--top-p", "0.9",
        "--repeat-penalty", "1.1",
        "--ctx-size", "4096",
        "-ngl", "0",
        "--log-disable"  # Disable logging for cleaner output
    ]
    
    try:
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            timeout=120  # 2 minute timeout
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"llama-cli failed: {result.stderr}")
        
        # Parse output to extract just the response
        output = result.stdout
        # Find the response after "### Response:\n"
        if "### Response:\n" in output:
            response = output.split("### Response:\n", 1)[1].strip()
        else:
            response = output.strip()
        
        return response
        
    except subprocess.TimeoutExpired:
        raise TimeoutError("Inference timed out")

@app.route("/health")
def health():
    """Health check"""
    return {"status": "healthy", "model": DEFAULT_MODEL}

@app.route("/v1/completions", methods=["POST"])
def completions():
    """OpenAI-style completions endpoint"""
    try:
        data = request.json
        prompt = data.get("prompt", "")
        max_tokens = data.get("max_tokens", 512)
        temperature = data.get("temperature", 0.2)
        model = data.get("model", DEFAULT_MODEL)
        
        if not prompt:
            return jsonify({"error": "No prompt provided"}), 400
        
        start_time = time.time()
        
        # Run inference
        model_file = os.path.join(MODELS_DIR, model) if not model.endswith('.gguf') else os.path.join(MODELS_DIR, f"{model}.gguf")
        response_text = run_inference(prompt, model_file, max_tokens, temperature)
        
        end_time = time.time()
        
        return jsonify({
            "id": f"cmpl-{int(time.time())}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "text": response_text,
                "index": 0,
                "logprobs": None,
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": len(prompt.split()),  # Rough estimate
                "completion_tokens": len(response_text.split()),  # Rough estimate
                "total_tokens": len(prompt.split()) + len(response_text.split())
            },
            "inference_time": end_time - start_time
        })
        
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except TimeoutError as e:
        return jsonify({"error": str(e)}), 504
    except Exception as e:
        return jsonify({"error": f"Internal error: {str(e)}"}), 500

@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions():
    """OpenAI-style chat completions endpoint"""
    try:
        data = request.json
        messages = data.get("messages", [])
        max_tokens = data.get("max_tokens", 512)
        temperature = data.get("temperature", 0.2)
        model = data.get("model", DEFAULT_MODEL)
        
        if not messages:
            return jsonify({"error": "No messages provided"}), 400
        
        # Convert messages to single prompt
        prompt_parts = []
        for msg in messages:
            if msg["role"] == "system":
                prompt_parts.append(f"System: {msg['content']}")
            elif msg["role"] == "user":
                prompt_parts.append(f"User: {msg['content']}")
            elif msg["role"] == "assistant":
                prompt_parts.append(f"Assistant: {msg['content']}")
        
        # Use last user message as the main prompt
        last_user_msg = None
        for msg in reversed(messages):
            if msg["role"] == "user":
                last_user_msg = msg["content"]
                break
        
        if not last_user_msg:
            return jsonify({"error": "No user message found"}), 400
        
        start_time = time.time()
        
        # Run inference
        model_file = os.path.join(MODELS_DIR, model) if not model.endswith('.gguf') else os.path.join(MODELS_DIR, f"{model}.gguf")
        response_text = run_inference(last_user_msg, model_file, max_tokens, temperature)
        
        end_time = time.time()
        
        return jsonify({
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_text
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": sum(len(m["content"].split()) for m in messages),
                "completion_tokens": len(response_text.split()),
                "total_tokens": sum(len(m["content"].split()) for m in messages) + len(response_text.split())
            },
            "inference_time": end_time - start_time
        })
        
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except TimeoutError as e:
        return jsonify({"error": str(e)}), 504
    except Exception as e:
        return jsonify({"error": f"Internal error: {str(e)}"}), 500

@app.route("/v1/models")
def models():
    """List available models"""
    try:
        model_files = []
        if os.path.exists(MODELS_DIR):
            model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith('.gguf')]
        
        return jsonify({
            "object": "list",
            "data": [{
                "id": f.replace('.gguf', ''),
                "object": "model",
                "created": int(os.path.getmtime(os.path.join(MODELS_DIR, f))),
                "owned_by": "aletheia"
            } for f in model_files]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("Starting Aletheia CPU Server...")
    print(f"Models directory: {MODELS_DIR}")
    print(f"Default model: {DEFAULT_MODEL}")
    print(f"llama-cli path: {LLAMA_CLI}")
    
    # Check if models exist
    if not os.path.exists(MODELS_DIR):
        print("Warning: Models directory not found. Run 'make cpu-convert' first.")
    
    app.run(host="0.0.0.0", port=8002, debug=False)