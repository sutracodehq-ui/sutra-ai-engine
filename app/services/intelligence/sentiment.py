"""
Sentiment Service — detects user tone and emotional state.

Software Factory: Uses run_pipeline("sentiment") for config-driven
LLM execution. ALL prompts, schemas, fallbacks, driver chains, and
tenant learning are defined in intelligence_config.yaml.

This file should NEVER need editing. To change behavior:
  → Edit intelligence_config.yaml → intelligence_pipelines → sentiment
"""

import logging

logger = logging.getLogger(__name__)


class SentimentService:
    """Service for real-time sentiment analysis.

    Schema-agnostic — returns whatever the YAML pipeline produces.
    Fallback response is defined in YAML (fallback_response key).
    Tenant learning is handled by run_pipeline() via tenant_id.
    """

    # Default fallback if YAML config has none — absolute last resort
    _SAFE_FALLBACK = {"score": 0.0, "label": "neutral", "vibe": "neutral"}

    @classmethod
    async def analyze(cls, text: str, tenant_id: int | None = None) -> dict:
        """
        Analyze the sentiment of the provided text.

        Pipeline: run_pipeline("sentiment") → parsed JSON dict
        On failure, returns fallback_response from YAML config.
        Tenant learning: stores result in Qdrant if tenant_id provided.
        """
        from app.lib.llm_pipeline import run_pipeline, get_pipeline_config

        if not text or len(text) < 5:
            return get_pipeline_config("sentiment").get("fallback_response", cls._SAFE_FALLBACK)

        result = await run_pipeline("sentiment", {"text": text}, tenant_id=tenant_id)

        # Pipeline returns fallback_response from YAML on total failure,
        # or None if no fallback is configured
        if result and isinstance(result, dict):
            return result

        return get_pipeline_config("sentiment").get("fallback_response", cls._SAFE_FALLBACK)
