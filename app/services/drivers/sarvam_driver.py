"""Sarvam AI Driver — Indian language-first LLM (sarvam-m, sarvam-2b).

Sarvam AI specializes in Indian languages with native understanding.
API is OpenAI-compatible at https://api.sarvam.ai/v1

Key strengths:
- Native Hindi/Hinglish + 10 Indic languages
- Lightweight for fast inference
- Translation + multilingual chat
"""

from typing import AsyncGenerator

from openai import AsyncOpenAI

from app.config import get_settings
from app.services.drivers.base import LlmDriver, LlmResponse


class SarvamDriver(LlmDriver):
    """Sarvam AI — Indian language models via OpenAI-compatible API."""

    def __init__(self):
        s = get_settings()
        self._client = AsyncOpenAI(
            api_key=s.sarvam_api_key,
            base_url="https://api.sarvam.ai/v1",
        )
        self._model = s.sarvam_model
        self._max_tokens = s.sarvam_max_tokens
        self._temperature = s.sarvam_temperature

    def name(self) -> str:
        return "sarvam"

    async def complete(self, system_prompt: str, user_prompt: str, **options) -> LlmResponse:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return await self.chat(messages, **options)

    async def chat(self, messages: list[dict], **options) -> LlmResponse:
        response = await self._client.chat.completions.create(
            model=options.get("model", self._model),
            messages=messages,
            max_tokens=options.get("max_tokens", self._max_tokens),
            temperature=options.get("temperature", self._temperature),
        )
        choice = response.choices[0]
        usage = response.usage

        return LlmResponse(
            content=choice.message.content or "",
            raw_response=choice.message.content or "",
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            total_tokens=usage.total_tokens if usage else 0,
            model=response.model,
            driver="sarvam",
        )

    async def stream(self, messages: list[dict], **options) -> AsyncGenerator[str, None]:
        stream = await self._client.chat.completions.create(
            model=options.get("model", self._model),
            messages=messages,
            max_tokens=options.get("max_tokens", self._max_tokens),
            temperature=options.get("temperature", self._temperature),
            stream=True,
        )
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
