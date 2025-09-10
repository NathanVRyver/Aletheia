#!/usr/bin/env python3
"""
OpenAI-compatible proxy server for Aletheia
Adds custom system prompts, logging, and rate limiting
"""
import asyncio
import json
import logging
import time
from typing import Dict, List, Optional
from datetime import datetime

import httpx
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Aletheia Proxy", description="OpenAI-compatible proxy for Aletheia model")
security = HTTPBearer()

# Configuration
VLLM_BASE_URL = "http://localhost:8000"
VLLM_API_KEY = "aletheia-key-123"

# System prompt for Aletheia
ALETHEIA_SYSTEM_PROMPT = """You are Aletheia, an expert AI assistant specializing in systems engineering, backend development, and technical problem-solving. 

Your expertise includes:
- Distributed systems design and architecture
- Database design and optimization
- Performance optimization and scalability
- DevOps and infrastructure
- Programming best practices
- System reliability and monitoring

Provide concise, technically accurate answers with practical examples and trade-offs when relevant. Focus on production-ready solutions and industry best practices."""

# Request/Response models
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "aletheia"
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.2
    stream: Optional[bool] = False

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict]
    usage: Dict

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple rate limiting (in production, use Redis)
request_counts = {}
RATE_LIMIT = 60  # requests per minute

def check_rate_limit(client_ip: str):
    """Simple rate limiting"""
    now = time.time()
    minute = int(now // 60)
    
    if client_ip not in request_counts:
        request_counts[client_ip] = {}
    
    if minute not in request_counts[client_ip]:
        request_counts[client_ip][minute] = 0
    
    # Clean old entries
    old_minutes = [m for m in request_counts[client_ip].keys() if m < minute - 1]
    for old_minute in old_minutes:
        del request_counts[client_ip][old_minute]
    
    if request_counts[client_ip][minute] >= RATE_LIMIT:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    request_counts[client_ip][minute] += 1

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify API key"""
    # In production, implement proper token verification
    if credentials.credentials != "aletheia-proxy-key":
        raise HTTPException(status_code=401, detail="Invalid API key")
    return credentials.credentials

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{VLLM_BASE_URL}/health", timeout=5.0)
            return {"status": "healthy", "vllm_status": response.status_code}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    client_request: Request,
    token: str = Depends(verify_token)
):
    """Chat completions endpoint with custom system prompt injection"""
    client_ip = client_request.client.host
    check_rate_limit(client_ip)
    
    # Log request
    logger.info(f"Chat request from {client_ip}: {len(request.messages)} messages")
    
    # Inject Aletheia system prompt if not present
    messages = request.messages.copy()
    has_system = any(msg.role == "system" for msg in messages)
    
    if not has_system:
        system_msg = ChatMessage(role="system", content=ALETHEIA_SYSTEM_PROMPT)
        messages.insert(0, system_msg)
    
    # Prepare request for vLLM
    vllm_request = {
        "model": "aletheia",
        "messages": [{"role": msg.role, "content": msg.content} for msg in messages],
        "max_tokens": request.max_tokens,
        "temperature": request.temperature,
        "stream": request.stream
    }
    
    # Forward to vLLM
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{VLLM_BASE_URL}/v1/chat/completions",
                json=vllm_request,
                headers={"Authorization": f"Bearer {VLLM_API_KEY}"}
            )
            
            if response.status_code != 200:
                logger.error(f"vLLM error: {response.status_code} - {response.text}")
                raise HTTPException(status_code=response.status_code, detail=response.text)
            
            result = response.json()
            
            # Log response
            if result.get("choices"):
                content_length = len(result["choices"][0]["message"]["content"])
                tokens_used = result.get("usage", {}).get("total_tokens", 0)
                logger.info(f"Response: {content_length} chars, {tokens_used} tokens")
            
            return result
            
    except httpx.TimeoutException:
        logger.error("Request to vLLM timed out")
        raise HTTPException(status_code=504, detail="Request timed out")
    except Exception as e:
        logger.error(f"Error forwarding to vLLM: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/v1/models")
async def list_models():
    """List available models"""
    return {
        "object": "list",
        "data": [
            {
                "id": "aletheia",
                "object": "model",
                "created": int(datetime.now().timestamp()),
                "owned_by": "aletheia-project"
            }
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)