"""Google Gemini LLM Driver — Flash, Pro, etc."""

from typing import AsyncGenerator

import google.generativeai as genai

from app.config import get_settings
from app.services.drivers.base import LlmDriver, LlmResponse


class GeminiDriver(LlmDriver):

    def __init__(self):
        s = get_settings()
        genai.configure(api_key=s.gemini_api_key)
        self._model_name = s.gemini_model
        self._max_tokens = s.gemini_max_tokens
        self._temperature = s.gemini_temperature

    def name(self) -> str:
        return "gemini"

    async def complete(self, system_prompt: str, user_prompt: str, **options) -> LlmResponse:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return await self.chat(messages, **options)

    async def chat(self, messages: list[dict], **options) -> LlmResponse:
        model_name = options.get("model", self._model_name)
        model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=next((m["content"] for m in messages if m["role"] == "system"), None),
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=options.get("max_tokens", self._max_tokens),
                temperature=options.get("temperature", self._temperature),
            ),
        )

        # Convert messages to Gemini format (user/model alternation)
        history = []
        for msg in messages:
            if msg["role"] == "system":
                continue
            role = "model" if msg["role"] == "assistant" else "user"
            history.append({"role": role, "parts": [msg["content"]]})

        # Use the last user message as the prompt
        chat_history = history[:-1] if len(history) > 1 else []
        last_msg = history[-1]["parts"][0] if history else ""

        chat = model.start_chat(history=chat_history)
        response = chat.send_message(last_msg)

        content = response.text or ""
        usage = response.usage_metadata

        return LlmResponse(
            content=content,
            raw_response=content,
            prompt_tokens=usage.prompt_token_count if usage else 0,
            completion_tokens=usage.candidates_token_count if usage else 0,
            total_tokens=usage.total_token_count if usage else 0,
            model=model_name,
            driver="gemini",
        )

    async def stream(self, messages: list[dict], **options) -> AsyncGenerator[str, None]:
        model_name = options.get("model", self._model_name)
        model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=next((m["content"] for m in messages if m["role"] == "system"), None),
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=options.get("max_tokens", self._max_tokens),
                temperature=options.get("temperature", self._temperature),
            ),
        )

        history = []
        for msg in messages:
            if msg["role"] == "system":
                continue
            role = "model" if msg["role"] == "assistant" else "user"
            history.append({"role": role, "parts": [msg["content"]]})

        chat_history = history[:-1] if len(history) > 1 else []
        last_msg = history[-1]["parts"][0] if history else ""

        chat = model.start_chat(history=chat_history)
        response = chat.send_message(last_msg, stream=True)

        for chunk in response:
            if chunk.text:
                yield chunk.text
