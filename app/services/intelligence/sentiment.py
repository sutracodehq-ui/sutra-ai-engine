"""
Sentiment Service — detects user tone and emotional state.

Software Factory: sentiment is a modular intelligence unit.
Use it to adjust AI personality, escalate to human, or skip
heavy-duty processing for frustrated users.

Uses cloud-first strategy for fast, accurate JSON extraction.
"""

import json
import logging
from typing import TypedDict

logger = logging.getLogger(__name__)


class SentimentResult(TypedDict):
    score: float  # -1.0 (angry) to 1.0 (delighted)
    label: str    # "angry", "frustrated", "neutral", "happy", "excited"
    vibe: str     # Short descriptive string of the user's tone


class SentimentService:
    """Service for real-time sentiment analysis."""

    _SYSTEM_PROMPT = "You are a sentiment analyzer. Be fast, accurate, and output raw JSON only."

    @classmethod
    async def analyze(cls, text: str) -> SentimentResult:
        """Analyze the sentiment of the provided text."""
        if not text or len(text) < 5:
            return {"score": 0.0, "label": "neutral", "vibe": "neutral"}

        prompt = f"""
Analyze the sentiment and tone of the following user message.
Return ONLY a JSON object with this schema:
{{
  "score": float (-1.0 to 1.0),
  "label": "angry" | "frustrated" | "neutral" | "happy" | "excited",
  "vibe": "string describing the vibe"
}}

USER MESSAGE: "{text}"
""".strip()

        try:
            data = await cls._call_cloud_first(prompt)
            if data:
                return {
                    "score": data.get("score", 0.0),
                    "label": data.get("label", "neutral"),
                    "vibe": data.get("vibe", "neutral"),
                }
        except Exception as e:
            logger.error(f"Sentiment Analysis failed: {e}")

        return {"score": 0.0, "label": "neutral", "vibe": "error_fallback"}

    @classmethod
    async def _call_cloud_first(cls, prompt: str) -> dict | None:
        """Cloud-first: Groq → Gemini → Anthropic → Ollama."""
        from app.services.intelligence.driver import get_driver_registry

        registry = get_driver_registry()

        for driver_name in ["groq", "gemini", "anthropic", "ollama"]:
            if not registry.circuit_breaker.is_available(driver_name):
                continue
            try:
                response = await registry.complete(
                    system_prompt=cls._SYSTEM_PROMPT,
                    user_prompt=prompt,
                    driver_override=driver_name,
                    temperature=0.0,
                    json_mode=True,
                )
                if response.content:
                    return json.loads(response.content)
            except Exception as e:
                logger.warning(f"SentimentService: {driver_name} failed: {e}")
                continue
        return None
