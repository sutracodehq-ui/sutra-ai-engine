"""OpenAI LLM Driver — GPT-4o, GPT-4o-mini, etc."""

import json
from typing import AsyncGenerator

from openai import AsyncOpenAI

from app.config import get_settings
from app.services.drivers.base import LlmDriver, LlmResponse


class OpenAiDriver(LlmDriver):

    def __init__(self):
        s = get_settings()
        self._client = AsyncOpenAI(api_key=s.openai_api_key, base_url="https://api.openai.com/v1")
        self._model = s.openai_model
        self._max_tokens = s.openai_max_tokens
        self._temperature = s.openai_temperature

    def name(self) -> str:
        return "openai"

    async def complete(self, system_prompt: str, user_prompt: str, **options) -> LlmResponse:
        # Check for images in options (Vision support)
        images = options.get("images", [])
        if images:
            user_content = [{"type": "text", "text": user_prompt}]
            for img in images:
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": img} if img.startswith("http") else {"url": f"data:image/jpeg;base64,{img}"}
                })
            user_msg = {"role": "user", "content": user_content}
        else:
            user_msg = {"role": "user", "content": user_prompt}

        messages = [
            {"role": "system", "content": system_prompt},
            user_msg,
        ]
        return await self.chat(messages, **options)

    async def chat(self, messages: list[dict], **options) -> LlmResponse:
        response = await self._client.chat.completions.create(
            model=options.get("model", self._model),
            messages=messages,
            max_tokens=options.get("max_tokens", self._max_tokens),
            temperature=options.get("temperature", self._temperature),
            response_format={"type": "json_object"} if options.get("json_mode") else None,
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
            driver="openai",
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
