"""
Language Service — detects input language and manages translations.

Software Factory: Uses run_pipeline("language_detect") and
run_pipeline("language_translate") for config-driven LLM execution.
All prompts, driver chains, and fallback responses are defined in
intelligence_config.yaml → intelligence_pipelines.
"""

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
        """
        Detect the language of the provided text.

        Pipeline: run_pipeline("language_detect") → parsed JSON dict
        On failure, returns English fallback from YAML config.
        """
        if not text or len(text) < 10:
            return {"code": "en", "name": "English", "confidence": 1.0}

        from app.lib.llm_pipeline import run_pipeline

        result = await run_pipeline("language_detect", {"text": text[:500]})

        if result and isinstance(result, dict):
            return {
                "code": result.get("code", "en"),
                "name": result.get("name", "English"),
                "confidence": float(result.get("confidence", 0.0)),
            }

        return {"code": "en", "name": "English", "confidence": 0.0}

    @classmethod
    async def translate(cls, text: str, target_lang: str) -> str:
        """
        Translate text to the target language code.

        Pipeline: run_pipeline("language_translate") → raw string
        On failure, returns original text unchanged.
        """
        from app.lib.llm_pipeline import run_pipeline

        result = await run_pipeline("language_translate", {
            "text": text,
            "target_lang": target_lang,
        })

        return result if isinstance(result, str) else text
