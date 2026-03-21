"""
Response Cache — Redis-based semantic cache for LLM responses.

Software Factory Principle: Efficiency through automation.

Skips the LLM entirely for identical or near-identical queries.
Uses a hash of (agent_type + prompt) as the cache key with configurable TTL.

Architecture:
    Request → Hash(agent + prompt) → Redis lookup
        → HIT: return cached response (0ms latency, 0 tokens)
        → MISS: call LLM → cache response → return
"""

import hashlib
import json
import logging
from typing import Optional

import yaml

from app.config import get_settings

logger = logging.getLogger(__name__)


def _load_cache_config() -> dict:
    """Load cache config from intelligence_config.yaml."""
    from pathlib import Path
    config_path = Path("intelligence_config.yaml")
    if not config_path.exists():
        return {}
    with open(config_path) as f:
        config = yaml.safe_load(f) or {}
    return config.get("response_cache", {})


class ResponseCache:
    """
    Redis-backed response cache with semantic key hashing.

    Features:
    - Exact match caching via content-hash
    - Per-agent TTL control
    - Cache hit/miss metrics
    - Automatic eviction via Redis TTL
    """

    def __init__(self):
        self._redis = None
        self._hits = 0
        self._misses = 0

    async def _get_redis(self):
        """Lazy-init async Redis client."""
        if self._redis is None:
            import redis.asyncio as aioredis
            settings = get_settings()
            self._redis = aioredis.from_url(
                settings.redis_url,
                decode_responses=True,
            )
        return self._redis

    def _make_key(self, agent_type: str, prompt: str, context_hash: str = "") -> str:
        """
        Create a cache key from agent type and prompt.
        
        Key format: cache:response:{hash}
        Hash input: agent_type + normalized_prompt + optional_context
        """
        # Normalize prompt: lowercase, strip whitespace, remove extra spaces
        normalized = " ".join(prompt.lower().strip().split())
        raw = f"{agent_type}:{normalized}:{context_hash}"
        content_hash = hashlib.sha256(raw.encode()).hexdigest()[:32]
        return f"cache:response:{content_hash}"

    async def get(self, agent_type: str, prompt: str, context_hash: str = "") -> Optional[dict]:
        """
        Check cache for an existing response.
        
        Returns cached response dict or None on miss.
        """
        config = _load_cache_config()
        if not config.get("enabled", True):
            return None

        try:
            redis = await self._get_redis()
            key = self._make_key(agent_type, prompt, context_hash)
            cached = await redis.get(key)

            if cached:
                self._hits += 1
                data = json.loads(cached)
                logger.info(
                    f"ResponseCache: HIT for {agent_type} "
                    f"(hits={self._hits}, misses={self._misses})"
                )
                return data

            self._misses += 1
            return None

        except Exception as e:
            logger.warning(f"ResponseCache: get failed: {e}")
            return None

    async def put(
        self,
        agent_type: str,
        prompt: str,
        response_content: str,
        metadata: dict | None = None,
        context_hash: str = "",
    ) -> bool:
        """
        Cache a response for future identical queries.
        """
        config = _load_cache_config()
        if not config.get("enabled", True):
            return False

        # Skip caching very short responses (likely errors)
        min_length = config.get("min_response_length", 50)
        if len(response_content) < min_length:
            return False

        try:
            redis = await self._get_redis()
            key = self._make_key(agent_type, prompt, context_hash)

            # Per-agent or global TTL
            agent_ttls = config.get("agent_ttl", {})
            ttl = agent_ttls.get(agent_type, config.get("default_ttl", 3600))

            data = {
                "content": response_content,
                "agent_type": agent_type,
                "cached": True,
                "metadata": metadata or {},
            }

            await redis.set(key, json.dumps(data, ensure_ascii=False), ex=ttl)
            logger.debug(f"ResponseCache: stored for {agent_type} (ttl={ttl}s)")
            return True

        except Exception as e:
            logger.warning(f"ResponseCache: put failed: {e}")
            return False

    async def invalidate(self, agent_type: str, prompt: str, context_hash: str = "") -> bool:
        """Invalidate a specific cached response."""
        try:
            redis = await self._get_redis()
            key = self._make_key(agent_type, prompt, context_hash)
            await redis.delete(key)
            return True
        except Exception as e:
            logger.warning(f"ResponseCache: invalidate failed: {e}")
            return False

    async def clear_agent(self, agent_type: str) -> int:
        """Clear all cached responses for a specific agent."""
        try:
            redis = await self._get_redis()
            # Use SCAN to find keys (pattern match isn't perfect but sufficient)
            cursor = 0
            deleted = 0
            while True:
                cursor, keys = await redis.scan(cursor, match="cache:response:*", count=100)
                for key in keys:
                    cached = await redis.get(key)
                    if cached:
                        data = json.loads(cached)
                        if data.get("agent_type") == agent_type:
                            await redis.delete(key)
                            deleted += 1
                if cursor == 0:
                    break
            logger.info(f"ResponseCache: cleared {deleted} entries for {agent_type}")
            return deleted
        except Exception as e:
            logger.warning(f"ResponseCache: clear failed: {e}")
            return 0

    def stats(self) -> dict:
        """Return cache hit/miss statistics."""
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "total": total,
            "hit_rate": round(self._hits / max(total, 1) * 100, 1),
        }


# ─── Singleton ──────────────────────────────────────────────
_cache: ResponseCache | None = None


def get_response_cache() -> ResponseCache:
    global _cache
    if _cache is None:
        _cache = ResponseCache()
    return _cache
