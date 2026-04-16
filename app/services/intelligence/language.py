"""
Language Service — detects input language and manages translations.

Software Factory: Uses run_pipeline("language_detect") and
run_pipeline("language_translate") for config-driven LLM execution.
ALL prompts, schemas, fallbacks, driver chains, and tenant learning
are defined in intelligence_config.yaml.

This file should NEVER need editing. To change behavior:
  → Edit intelligence_config.yaml → intelligence_pipelines → language_detect / language_translate
"""

import logging

logger = logging.getLogger(__name__)


class LanguageService:
    """Service for multilingual support and detection.

    Schema-agnostic — returns whatever the YAML pipeline produces.
    Fallback response is defined in YAML (fallback_response key).
    Tenant learning is handled by run_pipeline() via tenant_id.
    """

    # Absolute last resort if YAML config has no fallback
    _SAFE_FALLBACK = {"code": "en", "name": "English", "confidence": 0.0}

    @classmethod
    async def detect(cls, text: str, tenant_id: int | None = None) -> dict:
        """
        Detect the language of the provided text.

        Pipeline: run_pipeline("language_detect") → parsed JSON dict
        On failure, returns fallback_response from YAML config.
        Tenant learning: stores result in Qdrant if tenant_id provided.
        """
        from app.lib.llm_pipeline import run_pipeline, get_pipeline_config

        cfg = get_pipeline_config("language_detect")
        max_chars = cfg.get("max_content_chars", 500)
        fallback = cfg.get("fallback_response", cls._SAFE_FALLBACK)

        if not text or len(text) < 10:
            return fallback

        result = await run_pipeline(
            "language_detect", {"text": text[:max_chars]}, tenant_id=tenant_id
        )

        if result and isinstance(result, dict):
            return result

        return fallback

    @classmethod
    async def translate(cls, text: str, target_lang: str, tenant_id: int | None = None) -> str:
        """
        Translate text to the target language code.

        Pipeline: run_pipeline("language_translate") → raw string
        On failure, returns original text unchanged.
        """
        from app.lib.llm_pipeline import run_pipeline

        result = await run_pipeline(
            "language_translate",
            {"text": text, "target_lang": target_lang},
            tenant_id=tenant_id,
        )

        return result if isinstance(result, str) else text
