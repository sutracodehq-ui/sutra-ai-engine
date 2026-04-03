"""
Quality Engine — Unified quality scoring and tracking.

Software Factory Principle: Consolidation through polymorphism.

Merges two previously separate services:
    - QualityGate:    Multi-dimensional auto-scorer with retry prompt generation
    - QualityTracker: Redis-backed rolling average for routing decisions

Architecture:
    LLM Response → QualityEngine.score() → {total, dimensions, passed}
                 → QualityEngine.record() → Redis rolling average
    HybridRouter → QualityEngine.get_route_hint() → "fast_local" | "standard" | "direct_cloud"
"""

import json
import logging
from typing import Optional

from app.config import get_settings
from app.services.drivers.base import LlmResponse

logger = logging.getLogger(__name__)


class QualityEngine:
    """
    Unified quality scoring + rolling average tracking.

    Scoring dimensions (each 0-10):
    - completeness: does the response cover all expected fields?
    - format: is it valid JSON when expected?
    - length: is the response substantial enough?
    - coherence: basic structure and language quality checks

    Routing hints based on rolling average:
    - avg > 8.0 → fast path (trust local, skip quality gate)
    - avg < 5.0 → direct cloud (local isn't good enough yet)
    - 5.0-8.0 → standard path (try local → quality gate → maybe escalate)
    """

    WINDOW_SIZE = 20
    KEY_PREFIX = "hybrid:quality:"

    def __init__(self, *, enabled: bool = True, threshold: int = 6, max_retries: int = 1):
        self._enabled = enabled
        self._threshold = threshold
        self._max_retries = max_retries
        self._redis = None

    # ─── Redis (Lazy Init) ──────────────────────────────────

    async def _get_redis(self):
        """Lazy-init Redis connection."""
        if self._redis is None:
            import redis.asyncio as aioredis
            settings = get_settings()
            self._redis = aioredis.from_url(settings.celery_broker_url.replace("/1", "/2"))
        return self._redis

    # ─── Scoring (from QualityGate) ─────────────────────────

    def score(self, response: LlmResponse, expected_fields: list[str] | None = None) -> dict:
        """
        Score a response on multiple dimensions.
        """
        from app.services.intelligence.response_filter import get_response_filter
        refilter = get_response_filter()
        
        # Use ResponseFilter to get structured data (handles cleaning/parsing)
        result = refilter.filter(response.content or "", {"response_schema": expected_fields})
        content = response.content.strip()
        dimensions = {}

        dimensions["format"] = 1.0 if result.parsed else self._score_format_hint(content)
        dimensions["completeness"] = self._score_completeness_from_result(result, expected_fields)
        dimensions["length"] = self._score_length(content)
        dimensions["coherence"] = self._score_coherence(content)

        weights = {"format": 0.35, "completeness": 0.30, "length": 0.15, "coherence": 0.20}
        total = sum(dimensions[d] * weights[d] for d in dimensions) * 10

        passed = total >= self._threshold

        result = {
            "total": round(total, 1),
            "threshold": self._threshold,
            "passed": passed,
            "dimensions": {k: round(v * 10, 1) for k, v in dimensions.items()},
        }

        log_fn = logger.info if passed else logger.warning
        log_fn(f"QualityEngine: score={result['total']}/{self._threshold} passed={passed}")

        return result

    def _score_format_hint(self, content: str) -> float:
        """Score likely JSON-ness even if parsing failed (0.0-1.0)."""
        if "{" in content and "}" in content:
            return 0.5
        return 0.3

    def _score_completeness_from_result(self, result, expected_fields: list[str] | None) -> float:
        """Score field coverage from already-parsed result (0.0-1.0)."""
        if not expected_fields:
            return 0.8
        if not result.parsed or not isinstance(result.data, dict):
            return 0.3
        
        present = sum(1 for f in expected_fields if f in result.data)
        return present / len(expected_fields)

    def _score_length(self, content: str) -> float:
        """Score response length — penalize too short or empty."""
        word_count = len(content.split())
        if word_count == 0:
            return 0.0
        if word_count < 10:
            return 0.3
        if word_count < 30:
            return 0.6
        if word_count > 2000:
            return 0.8
        return 1.0

    def _score_coherence(self, content: str) -> float:
        """Basic coherence scoring."""
        score = 1.0

        error_signals = ["I cannot", "I'm sorry", "As an AI", "I don't have", "error", "exception"]
        for signal in error_signals:
            if signal.lower() in content.lower():
                score -= 0.2

        words = content.lower().split()
        if len(words) > 20:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.3:
                score -= 0.3

        return max(0.0, score)

    def build_retry_prompt(self, original_prompt: str, score_result: dict) -> str:
        """Build an augmented prompt for regeneration after quality failure."""
        weak_dimensions = [
            dim for dim, val in score_result["dimensions"].items() if val < 6.0
        ]

        augmentation = "\n\n--- IMPORTANT: QUALITY REQUIREMENTS ---\n"
        augmentation += f"Your previous response scored {score_result['total']}/10. "
        augmentation += "Please improve on these dimensions:\n"

        for dim in weak_dimensions:
            if dim == "format":
                augmentation += "- Respond with VALID JSON only. No markdown code fences.\n"
            elif dim == "completeness":
                augmentation += "- Include ALL required fields in your response.\n"
            elif dim == "length":
                augmentation += "- Provide more detailed, substantial content.\n"
            elif dim == "coherence":
                augmentation += "- Be more specific and actionable. Avoid generic AI disclaimers.\n"

        return original_prompt + augmentation

    # ─── Tracking (from QualityTracker) ─────────────────────

    async def record(self, agent_type: str, score: float) -> None:
        """Record a quality score for an agent (Redis rolling average)."""
        try:
            r = await self._get_redis()
            key = f"{self.KEY_PREFIX}{agent_type}"

            await r.lpush(key, str(score))
            await r.ltrim(key, 0, self.WINDOW_SIZE - 1)
            await r.expire(key, 86400 * 7)

        except Exception as e:
            logger.debug(f"QualityEngine record failed: {e}")

    async def get_average(self, agent_type: str) -> Optional[float]:
        """Get rolling average quality score for an agent."""
        try:
            r = await self._get_redis()
            key = f"{self.KEY_PREFIX}{agent_type}"

            scores = await r.lrange(key, 0, self.WINDOW_SIZE - 1)
            if not scores:
                return None

            values = [float(s) for s in scores]
            avg = sum(values) / len(values)
            return round(avg, 2)

        except Exception as e:
            logger.debug(f"QualityEngine get_average failed: {e}")
            return None

    async def get_route_hint(self, agent_type: str) -> str:
        """
        Get routing hint based on historical quality.

        Returns: "fast_local" | "standard" | "direct_cloud"
        """
        settings = get_settings()
        avg = await self.get_average(agent_type)

        if avg is None:
            return "standard"

        if avg >= settings.ai_hybrid_fast_path_threshold:
            return "fast_local"
        elif avg <= settings.ai_hybrid_direct_cloud_threshold:
            return "direct_cloud"
        return "standard"


# ─── Singleton ──────────────────────────────────────────────

_engine: QualityEngine | None = None


def get_quality_engine() -> QualityEngine:
    """Get the global QualityEngine singleton."""
    global _engine
    if _engine is None:
        _engine = QualityEngine()
    return _engine
