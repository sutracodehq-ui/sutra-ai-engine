"""
LLM Service — unified interface for all LLM operations.

Wraps the DriverManager and adds voice profile injection and fallback logic.
Software Factory: consumers use LlmService, never drivers directly.
"""

import logging
from typing import AsyncGenerator

from app.services.driver_manager import get_driver_manager
from app.services.drivers.base import LlmResponse

logger = logging.getLogger(__name__)


class LlmService:
    """Unified LLM interface with explicit driver/model control."""

    def __init__(self, manager=None):
        self._manager = manager or get_driver_manager()

    async def complete(
        self,
        prompt: str,
        system_prompt: str = "You are a helpful assistant.",
        messages: list[dict] | None = None,
        driver: str | None = None,
        model: str | None = None,
        **kwargs
    ) -> LlmResponse:
        """Single-turn completion with explicit driver/model overrides."""
        return await self._manager.complete(
            system_prompt,
            prompt,
            driver_override=driver,
            model_override=model,
            **kwargs
        )

    async def stream(
        self,
        prompt: str,
        system_prompt: str = "You are a helpful assistant.",
        messages: list[dict] | None = None,
        driver: str | None = None,
        model: str | None = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Streaming completion with explicit driver/model overrides."""
        # Convert single prompt to a messages list if driver prefers it
        # Most drivers in our system handle (system, user) correctly already.
        async for chunk in self._manager.stream(
            system_prompt,
            prompt,
            messages=messages,
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
