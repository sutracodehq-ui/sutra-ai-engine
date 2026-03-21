"""
Response Cache — Cache identical prompts to save LLM calls.

Engine Optimization: Same prompt + same agent = cached response.

Features:
- TTL-based cache (configurable per agent)
- Tenant-isolated (tenant A's cache ≠ tenant B's cache)
- Cache hit/miss tracking for analytics
- Max cache size with LRU eviction
- Cache bypass for real-time agents
"""

import hashlib
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """A cached response."""
    response: str
    agent_id: str
    prompt_hash: str
    created_at: float
    ttl: int
    hit_count: int = 0


class ResponseCache:
    """
    LRU cache for agent responses.
    
    Key: hash(tenant_id + agent_id + prompt)
    Value: cached response
    
    In production: Back with Redis for distributed caching.
    """

    # Default TTL per domain (seconds)
    DOMAIN_TTL = {
        "marketing": 3600,        # 1 hour (content is reusable)
        "ecommerce": 1800,        # 30 min (prices change)
        "finance": 900,           # 15 min (rates change)
        "health": 0,              # NO cache (safety-critical)
        "legal": 1800,            # 30 min
        "productivity": 600,      # 10 min
        "agriculture": 3600,      # 1 hour
        "real_estate": 3600,      # 1 hour
        "travel": 1800,           # 30 min
        "logistics": 300,         # 5 min (real-time)
        "government": 3600,       # 1 hour (stable data)
        "customer_success": 1800, # 30 min
    }

    # Agents that should NEVER be cached
    NO_CACHE_AGENTS = {
        "symptom_triage",       # Health — always fresh
        "mental_health_companion",  # Health — always personalized
        "medicine_info",        # Health — safety critical
        "dynamic_pricing",      # Changes constantly
        "shipment_tracker",     # Real-time data
        "daily_briefing",       # Daily = fresh
        "reminder_agent",       # Time-sensitive
    }

    def __init__(self, max_size: int = 10000, default_ttl: int = 1800):
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._hits = 0
        self._misses = 0

    def get(self, tenant_id: str, agent_id: str, prompt: str) -> str | None:
        """
        Get cached response.
        Returns None on miss or expired.
        """
        if agent_id in self.NO_CACHE_AGENTS:
            self._misses += 1
            return None

        key = self._key(tenant_id, agent_id, prompt)
        entry = self._cache.get(key)

        if not entry:
            self._misses += 1
            return None

        # Check TTL
        if time.time() - entry.created_at > entry.ttl:
            del self._cache[key]
            self._misses += 1
            return None

        # Cache hit
        entry.hit_count += 1
        self._hits += 1
        self._cache.move_to_end(key)

        logger.debug(f"Cache HIT: {agent_id} (hits={entry.hit_count})")
        return entry.response

    def set(
        self,
        tenant_id: str,
        agent_id: str,
        prompt: str,
        response: str,
        domain: str = "",
    ):
        """Cache a response."""
        if agent_id in self.NO_CACHE_AGENTS:
            return

        ttl = self.DOMAIN_TTL.get(domain, self._default_ttl)
        if ttl == 0:
            return

        key = self._key(tenant_id, agent_id, prompt)

        # Evict LRU if full
        while len(self._cache) >= self._max_size:
            self._cache.popitem(last=False)

        self._cache[key] = CacheEntry(
            response=response,
            agent_id=agent_id,
            prompt_hash=key[:16],
            created_at=time.time(),
            ttl=ttl,
        )

    def invalidate(self, tenant_id: str, agent_id: str | None = None):
        """Invalidate cache for a tenant (optionally for specific agent)."""
        keys_to_remove = [
            k for k, v in self._cache.items()
            if k.startswith(self._tenant_prefix(tenant_id))
            and (agent_id is None or v.agent_id == agent_id)
        ]
        for k in keys_to_remove:
            del self._cache[k]

        logger.info(f"Cache invalidated: {len(keys_to_remove)} entries for '{tenant_id}'")

    def stats(self) -> dict:
        """Cache performance stats."""
        total = self._hits + self._misses
        return {
            "size": len(self._cache),
            "max_size": self._max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": f"{(self._hits / max(total, 1)) * 100:.1f}%",
            "total_requests": total,
        }

    def _key(self, tenant_id: str, agent_id: str, prompt: str) -> str:
        raw = f"{tenant_id}:{agent_id}:{prompt.strip().lower()}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def _tenant_prefix(self, tenant_id: str) -> str:
        return hashlib.sha256(f"{tenant_id}:".encode()).hexdigest()[:8]


# ─── Singleton ──────────────────────────────────────────────

_cache: ResponseCache | None = None


def get_response_cache() -> ResponseCache:
    global _cache
    if _cache is None:
        _cache = ResponseCache()
    return _cache
