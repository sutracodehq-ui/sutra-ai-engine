"""
Moderation Service — content safety and guardrails.

Shield-AI: Ensures all inputs and outputs adhere to safety policies.
Integration: OpenAI Moderation API (fast, free for OpenAI users).
"""

import logging
from typing import TypedDict, List

import httpx
from app.config import get_settings

logger = logging.getLogger(__name__)


class ModerationResult(TypedDict):
    flagged: bool
    categories: List[str]
    score: float


class ModerationService:
    """Service to detect unsafe content in prompts or responses."""

    @classmethod
    async def check(cls, text: str) -> ModerationResult:
        """Check if content is safe using OpenAI's moderation endpoint."""
        if not text:
            return {"flagged": False, "categories": [], "score": 0.0}

        settings = get_settings()
        if not settings.openai_api_key:
            logger.warning("ModerationService: OpenAI API key missing. Skipping check.")
            return {"flagged": False, "categories": [], "score": 0.0}

        url = "https://api.openai.com/v1/moderations"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {settings.openai_api_key}",
        }
        payload = {"input": text}

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
                
                result = data["results"][0]
                flagged = result["flagged"]
                categories = [cat for cat, val in result["categories"].items() if val]
                
                # We calculate a max score across all categories
                score = max(result["category_scores"].values()) if "category_scores" in result else 0.0

                return {
                    "flagged": flagged,
                    "categories": categories,
                    "score": score
                }
        except Exception as e:
            logger.error(f"Moderation check failed: {e}")
            # In case of failure, we fail-safe (unflagged) but log the error
            return {"flagged": False, "categories": [], "score": 0.0}
