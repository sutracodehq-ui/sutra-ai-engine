"""
Quality Tracker — Redis-backed per-agent quality score tracker.

Tracks rolling average of local model quality scores per agent_type.
Used by HybridRouter to decide routing strategy:
- avg > 8.0 → fast path (trust local, skip quality gate)
- avg < 5.0 → direct cloud (local isn't good enough yet)
- 5.0-8.0 → standard path (try local → quality gate → maybe escalate)
"""

import logging
from typing import Optional

from app.config import get_settings

logger = logging.getLogger(__name__)


class QualityTracker:
    """
    Tracks per-agent local model quality using a Redis sorted set.
    
    Stores the last N quality scores per agent and computes
    a rolling average for routing decisions.
    """

    WINDOW_SIZE = 20  # Rolling window of last N scores
    KEY_PREFIX = "hybrid:quality:"

    def __init__(self):
        self._redis = None

    async def _get_redis(self):
        """Lazy-init Redis connection."""
        if self._redis is None:
            import redis.asyncio as aioredis
            settings = get_settings()
            self._redis = aioredis.from_url(settings.celery_broker_url.replace("/1", "/2"))
        return self._redis

    async def record(self, agent_type: str, score: float) -> None:
        """Record a quality score for an agent."""
        try:
            r = await self._get_redis()
            key = f"{self.KEY_PREFIX}{agent_type}"

            # Push score to list, trim to window size
            await r.lpush(key, str(score))
            await r.ltrim(key, 0, self.WINDOW_SIZE - 1)
            await r.expire(key, 86400 * 7)  # TTL 7 days

        except Exception as e:
            logger.debug(f"QualityTracker record failed: {e}")

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
            logger.debug(f"QualityTracker get_average failed: {e}")
            return None

    async def get_route_hint(self, agent_type: str) -> str:
        """
        Get routing hint based on historical quality.

        Returns: "fast_local" | "standard" | "direct_cloud"
        """
        settings = get_settings()
        avg = await self.get_average(agent_type)

        if avg is None:
            return "standard"  # No history, use standard path

        if avg >= settings.ai_hybrid_fast_path_threshold:
            return "fast_local"
        elif avg <= settings.ai_hybrid_direct_cloud_threshold:
            return "direct_cloud"
        return "standard"


# ─── Singleton ──────────────────────────────────────────
_tracker: QualityTracker | None = None


def get_quality_tracker() -> QualityTracker:
    global _tracker
    if _tracker is None:
        _tracker = QualityTracker()
    return _tracker
