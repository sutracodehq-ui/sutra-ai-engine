"""
Sentiment Service — detects user tone and emotional state.

Software Factory: sentiment is a modular intelligence unit. 
Use it to adjust AI personality, escalate to human, or skip 
heavy-duty processing for frustrated users.
"""

import logging
from typing import TypedDict

from app.services.llm_service import get_llm_service
from app.config import get_settings

logger = logging.getLogger(__name__)


class SentimentResult(TypedDict):
    score: float  # -1.0 (angry) to 1.0 (delighted)
    label: str    # "angry", "frustrated", "neutral", "happy", "excited"
    vibe: str     # Short descriptive string of the user's tone


class SentimentService:
    """Service for real-time sentiment analysis."""

    @classmethod
    async def analyze(cls, text: str) -> SentimentResult:
        """Analyze the sentiment of the provided text."""
        if not text or len(text) < 5:
            return {"score": 0.0, "label": "neutral", "vibe": "neutral"}

        settings = get_settings()
        service = get_llm_service()
        
        # We use a fast model for 'reflexive' intelligence like sentiment
        model = settings.ai_meta_prompt_model  # Default to Flash
        
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
            result = await service.complete(
                prompt=prompt,
                system_prompt="You are a sentiment analyzer. Be fast, accurate, and output raw JSON only.",
                model=model,
                temperature=0.0,
                json_mode=True
            )
            
            import json
            data = json.loads(result.get("content", "{}"))
            return {
                "score": data.get("score", 0.0),
                "label": data.get("label", "neutral"),
                "vibe": data.get("vibe", "neutral")
            }
        except Exception as e:
            logger.error(f"Sentiment Analysis failed: {e}")
            return {"score": 0.0, "label": "neutral", "vibe": "error_fallback"}
