import asyncio
import json
import logging
import os
import subprocess
import time
import uuid
from typing import List, Optional, Union

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("bitnet-server")

app = FastAPI(title="BitNet 1.58-bit Inference Server")

# --- Config ---
BITNET_CLI = os.environ.get("BITNET_CLI", "/opt/bitnet/build/bin/llama-cli")
BITNET_MODEL = os.environ.get("BITNET_MODEL", "/opt/bitnet/models/BitNet-b1.58-2B-4T.gguf")
DEFAULT_MAX_TOKENS = 512

# --- Schemas ---
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "bitnet-2b"
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = DEFAULT_MAX_TOKENS
    stream: Optional[bool] = False

class ChatCompletionResponseChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"

class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4()}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: dict = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

class ScoreRequest(BaseModel):
    prompt: str

class ScoreResponse(BaseModel):
    score: int
    complexity: str  # simple | moderate | complex

# --- Core Inference Logic ---
async def run_bitnet_inference(prompt: str, max_tokens: int) -> str:
    """Runs the bitnet-cli as a subprocess and captures output."""
    try:
        # Construct command for bitnet-cli
        # Note: We use -p for prompt and -n for max tokens.
        cmd = [
            BITNET_CLI,
            "-m", BITNET_MODEL,
            "-p", prompt,
            "-n", str(max_tokens),
            "--temp", "0.1",          # Low temperature for deterministic scoring
            "--repeat-penalty", "1.1" # Prevent loops
        ]
        
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            logger.error(f"BitNet CLI Error: {stderr.decode()}")
            raise HTTPException(status_code=500, detail="Inference engine failure")
            
        return stdout.decode().strip()
    except Exception as e:
        logger.error(f"Failed to run BitNet: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- Routes ---
@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    # 1. Simplistic Prompt Assembly
    # For 1-bit models, we usually just join the content for now
    full_prompt = ""
    for msg in request.messages:
        if msg.role == "system":
            full_prompt += f"System: {msg.content}\n"
        elif msg.role == "user":
            full_prompt += f"User: {msg.content}\n"
    full_prompt += "Assistant: "

    # 2. Run Inference
    logger.info(f"Running BitNet inference for prompt length: {len(full_prompt)}")
    content = await run_bitnet_inference(full_prompt, request.max_tokens or DEFAULT_MAX_TOKENS)

    # 3. Format Response
    return ChatCompletionResponse(
        model=request.model,
        choices=[
            ChatCompletionResponseChoice(
                message=ChatMessage(role="assistant", content=content)
            )
        ]
    )

@app.post("/v1/score")
async def score_complexity(request: ScoreRequest):
    """
    Score prompt complexity 1-5 in ~200ms.
    Used by SmartRouter for O(1)-like routing but with actual model insight.
    """
    # Truncate prompt to save time — we only need initial context for classification
    clean_prompt = request.prompt[:512].replace('"', '\"')
    
    # Precise, few-shot-like prompt for the 2B model
    grading_prompt = f"""Task: Rate prompt complexity 1-5.
Scale: 1=Greeting, 2=Fact, 3=Reasoning, 4=Analysis, 5=Coding/Complex
Rules: Reply with ONLY the number.

Prompt: {clean_prompt}
Score:"""

    logger.info(f"Scoring complexity for prompt (len: {len(request.prompt)})")
    
    # Run with very small max_tokens since we only need 1 digit
    result = await run_bitnet_inference(grading_prompt, max_tokens=10)
    
    # Extract first digit
    import re
    match = re.search(r'(\d)', result)
    score = int(match.group(1)) if match else 3  # Default to moderate
    
    # Bounds check
    score = max(1, min(5, score))
    
    # Map to tier
    if score <= 2:
        tier = "simple"
    elif score == 3:
        tier = "moderate"
    else:
        tier = "complex"
        
    logger.info(f"Result: Score {score} -> Tier: {tier}")
    
    return ScoreResponse(score=score, complexity=tier)

@app.get("/health")
async def health():
    if os.path.exists(BITNET_CLI) and os.path.exists(BITNET_MODEL):
        return {"status": "healthy", "engine": "bitnet.cpp", "model": "bitnet-2b"}
    return {"status": "degraded", "error": "binary or model missing"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8081)
