"""Anthropic LLM Driver — Claude Sonnet, Haiku, etc."""

from typing import AsyncGenerator

from anthropic import AsyncAnthropic

from app.config import get_settings
from app.services.drivers.base import LlmDriver, LlmResponse


class AnthropicDriver(LlmDriver):

    def __init__(self):
        s = get_settings()
        self._client = AsyncAnthropic(api_key=s.anthropic_api_key)
        self._model = s.anthropic_model
        self._max_tokens = s.anthropic_max_tokens
        self._temperature = s.anthropic_temperature

    def name(self) -> str:
        return "anthropic"

    async def complete(self, system_prompt: str, user_prompt: str, **options) -> LlmResponse:
        messages = [{"role": "user", "content": user_prompt}]
        return await self._call(messages, system_prompt, **options)

    async def chat(self, messages: list[dict], **options) -> LlmResponse:
        # Extract system message (Anthropic uses a separate `system` param)
        system = ""
        chat_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system = msg["content"]
            else:
                chat_messages.append(msg)
        return await self._call(chat_messages, system, **options)

    async def _call(self, messages: list[dict], system: str, **options) -> LlmResponse:
        response = await self._client.messages.create(
            model=options.get("model", self._model),
            max_tokens=options.get("max_tokens", self._max_tokens),
            temperature=options.get("temperature", self._temperature),
            system=system,
            messages=messages,
        )
        content = response.content[0].text if response.content else ""

        return LlmResponse(
            content=content,
            raw_response=content,
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
            total_tokens=response.usage.input_tokens + response.usage.output_tokens,
            model=response.model,
            driver="anthropic",
        )

    async def stream(self, messages: list[dict], **options) -> AsyncGenerator[str, None]:
        system = ""
        chat_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system = msg["content"]
            else:
                chat_messages.append(msg)

        async with self._client.messages.stream(
            model=options.get("model", self._model),
            max_tokens=options.get("max_tokens", self._max_tokens),
            temperature=options.get("temperature", self._temperature),
            system=system,
            messages=chat_messages,
        ) as stream:
            async for text in stream.text_stream:
                yield text
