"""
LLM Service — unified interface for all LLM operations.

Wraps the DriverManager and adds voice profile injection and fallback logic.
Software Factory: consumers use LlmService, never drivers directly.
"""

import logging
from typing import AsyncGenerator

from app.services.intelligence.driver import get_driver_registry
from app.services.drivers.base import LlmResponse

logger = logging.getLogger(__name__)


class LlmService:
    """Unified LLM interface with explicit driver/model control."""

    def __init__(self, registry=None):
        self._registry = registry or get_driver_registry()

    async def complete(
        self,
        prompt: str,
        system_prompt: str = "You are a helpful assistant.",
        driver: str | None = None,
        model: str | None = None,
        **kwargs
    ) -> LlmResponse:
        """Single-turn completion via hardened registry."""
        if "messages" in kwargs and kwargs["messages"] is not None:
            messages = kwargs.pop("messages")
            merged = []
            merged.extend(messages)
            
            if system_prompt and not any(m["role"] == "system" for m in merged):
                merged.insert(0, {"role": "system", "content": system_prompt})
                
            if prompt:
                merged.append({"role": "user", "content": prompt})
                
            return await self._registry.chat(
                merged,
                driver_override=driver,
                model_override=model,
                **kwargs
            )
            
        return await self._registry.complete(
            system_prompt=system_prompt,
            user_prompt=prompt,
            driver_override=driver,
            model_override=model,
            **kwargs
        )

    async def chat(
        self,
        messages: list[dict],
        system_prompt: str | None = None,
        driver: str | None = None,
        model: str | None = None,
        **kwargs
    ) -> LlmResponse:
        """Chat completion via hardened registry."""
        if system_prompt and not any(m["role"] == "system" for m in messages):
            messages.insert(0, {"role": "system", "content": system_prompt})
            
        return await self._registry.chat(
            messages,
            driver_override=driver,
            model_override=model,
            **kwargs
        )

    async def stream(
        self,
        prompt: str | None = None,
        system_prompt: str = "You are a helpful assistant.",
        messages: list[dict] | None = None,
        driver: str | None = None,
        model: str | None = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Streaming completion via hardened registry."""
        merged = []
        if messages is not None:
            merged.extend(messages)
            
        if system_prompt and not any(m["role"] == "system" for m in merged):
            merged.insert(0, {"role": "system", "content": system_prompt})
            
        if prompt:
            merged.append({"role": "user", "content": prompt})

        async for chunk in self._registry.stream(
            system_prompt=None,
            user_prompt=None,
            messages=merged,
            driver_override=driver,
            model_override=model,
            **kwargs
        ):
            yield chunk


# ─── Singleton ──────────────────────────────────────────────────

_service: LlmService | None = None


def get_llm_service() -> LlmService:
    """Get the global LLM service instance."""
    global _service
    if _service is None:
        _service = LlmService()
    return _service
