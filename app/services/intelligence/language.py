"""
Language Service — detects input language and manages translations.

Enables 'Native Language' support where the AI automatically
responds in the user's tongue without explicit instructions.

Uses cloud-first strategy for fast, accurate JSON extraction.
"""

import json
import logging
from typing import TypedDict

logger = logging.getLogger(__name__)


class LanguageResult(TypedDict):
    code: str      # ISO 639-1 code (en, es, hi, etc.)
    name: str      # English, Spanish, Hindi, etc.
    confidence: float


class LanguageService:
    """Service for multilingual support and detection."""

    @classmethod
    async def detect(cls, text: str) -> LanguageResult:
        """Detect the language of the provided text."""
        if not text or len(text) < 10:
            return {"code": "en", "name": "English", "confidence": 1.0}

        prompt = f"""
Identify the language of the text below.
Return ONLY a JSON object with:
{{
  "code": "iso_639_1_code",
  "name": "Full Language Name",
  "confidence": float (0.0 to 1.0)
}}

TEXT: "{text[:500]}"
""".strip()

        try:
            data = await cls._call_cloud_first(
                prompt,
                "You are a language detector. Output raw JSON ONLY.",
            )
            if data:
                return {
                    "code": data.get("code", "en"),
                    "name": data.get("name", "English"),
                    "confidence": data.get("confidence", 0.0),
                }
        except Exception as e:
            logger.error(f"Language Detection failed: {e}")

        return {"code": "en", "name": "English", "confidence": 0.0}

    @classmethod
    async def translate(cls, text: str, target_lang: str) -> str:
        """Translate text to the target language code."""
        prompt = f"Translate the following to {target_lang}:\n\n{text}"

        try:
            from app.services.intelligence.driver import get_driver_registry

            registry = get_driver_registry()

            for driver_name in ["groq", "gemini", "anthropic", "ollama"]:
                if not registry.circuit_breaker.is_available(driver_name):
                    continue
                try:
                    response = await registry.complete(
                        system_prompt="You are a professional translator. Only return the translated text.",
                        user_prompt=prompt,
                        driver_override=driver_name,
                        temperature=0.3,
                    )
                    if response.content:
                        return response.content
                except Exception:
                    continue
        except Exception as e:
            logger.error(f"Translation failed: {e}")

        return text  # Return original if all drivers fail

    @classmethod
    async def _call_cloud_first(cls, prompt: str, system_prompt: str) -> dict | None:
        """Cloud-first: Groq → Gemini → Anthropic → Ollama."""
        from app.services.intelligence.driver import get_driver_registry

        registry = get_driver_registry()

        for driver_name in ["groq", "gemini", "anthropic", "ollama"]:
            if not registry.circuit_breaker.is_available(driver_name):
                continue
            try:
                response = await registry.complete(
                    system_prompt=system_prompt,
                    user_prompt=prompt,
                    driver_override=driver_name,
                    temperature=0.0,
                    json_mode=True,
                )
                if response.content:
                    return json.loads(response.content)
            except Exception as e:
                logger.warning(f"LanguageService: {driver_name} failed: {e}")
                continue
        return None
