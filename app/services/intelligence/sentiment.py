"""
Sentiment Service — detects user tone and emotional state.

Software Factory: Uses run_pipeline("sentiment") for config-driven
LLM execution. All prompts, driver chains, and fallback responses
are defined in intelligence_config.yaml → intelligence_pipelines.
"""

import logging
from typing import TypedDict

logger = logging.getLogger(__name__)


class SentimentResult(TypedDict):
    score: float  # -1.0 (angry) to 1.0 (delighted)
    label: str    # "angry", "frustrated", "neutral", "happy", "excited"
    vibe: str     # Short descriptive string of the user's tone


class SentimentService:
    """Service for real-time sentiment analysis."""

    @classmethod
    async def analyze(cls, text: str) -> SentimentResult:
        """
        Analyze the sentiment of the provided text.

        Pipeline: run_pipeline("sentiment") → parsed JSON dict
        On failure, returns neutral fallback from YAML config.
        """
        if not text or len(text) < 5:
            return {"score": 0.0, "label": "neutral", "vibe": "neutral"}

        from app.lib.llm_pipeline import run_pipeline

        result = await run_pipeline("sentiment", {"text": text})

        if result and isinstance(result, dict):
            return {
                "score": float(result.get("score", 0.0)),
                "label": result.get("label", "neutral"),
                "vibe": result.get("vibe", "neutral"),
            }

        # Fallback is returned by run_pipeline from YAML config,
        # but if even that is None, return hardcoded safe default
        return {"score": 0.0, "label": "neutral", "vibe": "error_fallback"}
