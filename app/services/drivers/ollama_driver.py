"""Ollama LLM Driver — local open-source model inference (zero cost)."""

import json
from typing import AsyncGenerator

import httpx

from app.config import get_settings
from app.services.drivers.base import LlmDriver, LlmResponse


class OllamaDriver(LlmDriver):
    """HTTP-based driver for the Ollama local inference server."""

    def __init__(self):
        s = get_settings()
        self._base_url = s.ollama_base_url.rstrip("/")
        self._model = s.ollama_model
        self._max_tokens = s.ollama_max_tokens
        self._temperature = s.ollama_temperature

    def name(self) -> str:
        return "ollama"

    async def complete(self, system_prompt: str, user_prompt: str, **options) -> LlmResponse:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return await self.chat(messages, **options)

    async def chat(self, messages: list[dict], **options) -> LlmResponse:
        model = options.get("model", self._model)
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "num_predict": options.get("max_tokens", self._max_tokens),
                "temperature": options.get("temperature", self._temperature),
            },
        }

        # Split timeout: fail fast on connect (dead Ollama), generous read for inference
        timeout = httpx.Timeout(connect=10.0, read=90.0, write=10.0, pool=5.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(f"{self._base_url}/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()

        content = data.get("message", {}).get("content", "")

        return LlmResponse(
            content=content,
            raw_response=content,
            prompt_tokens=data.get("prompt_eval_count", 0),
            completion_tokens=data.get("eval_count", 0),
            total_tokens=data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
            model=model,
            driver="ollama",
        )

    async def stream(self, messages: list[dict], **options) -> AsyncGenerator[str, None]:
        model = options.get("model", self._model)
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "options": {
                "num_predict": options.get("max_tokens", self._max_tokens),
                "temperature": options.get("temperature", self._temperature),
            },
        }

        # Long read timeout for LLM inference (thinking before first token)
        # Short connect timeout to fail fast if Ollama is down
        timeout = httpx.Timeout(connect=10.0, read=300.0, write=10.0, pool=10.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream("POST", f"{self._base_url}/api/chat", json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if line.strip():
                        data = json.loads(line)
                        content = data.get("message", {}).get("content", "")
                        if content:
                            yield content
