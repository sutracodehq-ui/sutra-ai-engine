"""
LLM Service — unified interface for all LLM operations.

Wraps the DriverManager and adds voice profile injection, thinking middleware,
and convenience methods. This is what agents call.
"""

import logging
from typing import AsyncGenerator

from app.models.voice_profile import VoiceProfile
from app.services.driver_manager import DriverManager, get_driver_manager
from app.services.drivers.base import LlmResponse

logger = logging.getLogger(__name__)


class LlmService:
    """Unified LLM interface with voice injection and fallback."""

    def __init__(self, manager: DriverManager | None = None):
        self._manager = manager or get_driver_manager()

    async def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        voice: VoiceProfile | None = None,
        **options,
    ) -> LlmResponse:
        """Single-turn completion with optional voice profile."""
        if voice:
            system_prompt += "\n\n--- Voice Profile ---\n" + voice.to_system_prompt_modifier()
        return await self._manager.complete(system_prompt, user_prompt, **options)

    async def chat(
        self,
        messages: list[dict],
        voice: VoiceProfile | None = None,
        **options,
    ) -> LlmResponse:
        """Multi-turn chat with optional voice profile injection."""
        messages = self._inject_voice(messages, voice)
        return await self._manager.chat(messages, **options)

    async def stream(
        self,
        messages: list[dict],
        voice: VoiceProfile | None = None,
        **options,
    ) -> AsyncGenerator[str, None]:
        """Streaming chat completion."""
        messages = self._inject_voice(messages, voice)
        async for chunk in self._manager.stream(messages, **options):
            yield chunk

    def get_driver_name(self) -> str:
        """Get the primary driver name."""
        from app.config import get_settings
        return get_settings().ai_driver

    def _inject_voice(self, messages: list[dict], voice: VoiceProfile | None) -> list[dict]:
        """Inject voice profile into the system message."""
        if not voice:
            return messages

        modifier = "\n\n--- Voice Profile ---\n" + voice.to_system_prompt_modifier()
        messages = [m.copy() for m in messages]  # Don't mutate originals
        for msg in messages:
            if msg["role"] == "system":
                msg["content"] += modifier
                break
        return messages


# ─── Singleton ──────────────────────────────────────────────────

_service: LlmService | None = None


def get_llm_service() -> LlmService:
    global _service
    if _service is None:
        _service = LlmService()
    return _service
