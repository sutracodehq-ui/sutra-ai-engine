"""Groq LLM Driver — Llama 3.3 70B on Groq inference hardware."""

from typing import AsyncGenerator

from openai import AsyncOpenAI

from app.config import get_settings
from app.services.drivers.base import LlmDriver, LlmResponse


class GroqDriver(LlmDriver):
    """Uses OpenAI-compatible API via Groq's endpoint."""

    def __init__(self):
        s = get_settings()
        self._client = AsyncOpenAI(api_key=s.groq_api_key, base_url="https://api.groq.com/openai/v1")
        self._model = s.groq_model
        self._max_tokens = s.groq_max_tokens
        self._temperature = s.groq_temperature

    def name(self) -> str:
        return "groq"

    async def complete(self, system_prompt: str, user_prompt: str, **options) -> LlmResponse:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        options.pop("messages", None)
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
            driver="groq",
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
