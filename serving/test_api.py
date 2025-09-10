#!/usr/bin/env python3
"""
Test script for Aletheia API server
"""
import requests
import json
import sys

def test_api(host="localhost", port=8000, api_key="aletheia-key-123"):
    """Test the vLLM OpenAI-compatible API"""
    base_url = f"http://{host}:{port}"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Test data
    test_prompts = [
        "Explain the difference between epoll and kqueue.",
        "How do you implement a circuit breaker in microservices?",
        "What are the best practices for database connection pooling?"
    ]
    
    print(f"Testing Aletheia API at {base_url}")
    print("=" * 50)
    
    # Test health endpoint
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        print(f"Health check: {response.status_code}")
    except Exception as e:
        print(f"Health check failed: {e}")
        return False
    
    # Test chat completions
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n--- Test {i}/{len(test_prompts)} ---")
        print(f"Prompt: {prompt}")
        
        payload = {
            "model": "aletheia",
            "messages": [
                {
                    "role": "system",
                    "content": "You are Aletheia, an expert in systems engineering and backend development. Provide concise, technical answers."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            "max_tokens": 300,
            "temperature": 0.2,
            "stream": False
        }
        
        try:
            response = requests.post(
                f"{base_url}/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                print(f"Response: {content[:200]}{'...' if len(content) > 200 else ''}")
                print(f"Tokens: {result['usage']['total_tokens']}")
            else:
                print(f"Error {response.status_code}: {response.text}")
                
        except Exception as e:
            print(f"Request failed: {e}")
    
    print("\n" + "=" * 50)
    print("API testing complete!")
    return True

def main():
    """Main function"""
    host = sys.argv[1] if len(sys.argv) > 1 else "localhost"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
    
    success = test_api(host, port)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()