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
        # Check for images in options (Vision support)
        images = options.get("images", [])
        if images:
            # For Gemini, we combine text and images into the same prompt part
            parts = [user_prompt]
            for img in images:
                if img.startswith("http"):
                    # We would need to fetch the image or use the URL if Gemini API supports it via specific models
                    # For now, we assume base64 or pass as metadata
                    parts.append({"mime_type": "image/jpeg", "data": img}) 
                else:
                    parts.append({"mime_type": "image/jpeg", "data": img})
            user_msg = {"role": "user", "parts": parts}
        else:
            user_msg = {"role": "user", "parts": [user_prompt]}

        messages = [
            {"role": "system", "content": system_prompt},
            user_msg,
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

        history = []
        for msg in messages:
            if msg["role"] == "system":
                continue
            role = "model" if msg["role"] == "assistant" else "user"
            
            # parts can be a list or a single string
            parts = msg.get("parts") or [msg["content"]]
            history.append({"role": role, "parts": parts})

        chat_history = history[:-1] if len(history) > 1 else []
        last_parts = history[-1]["parts"] if history else [""]

        chat = model.start_chat(history=chat_history)
        response = chat.send_message(last_parts)

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
