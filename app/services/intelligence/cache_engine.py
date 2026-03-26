"""
Cache Engine — Unified caching with strategy pattern.

Software Factory Principle: Consolidation through polymorphism.

Merges three previously separate caches into one:
    - exact:    Redis-based, SHA-256(tenant+system+user+messages) — identical prompts (15-25% hit rate)
    - hash:     Redis-based, SHA-256(agent+prompt) — agent-level dedup with per-agent TTL
    - semantic: ChromaDB-based, cosine similarity — catches paraphrased prompts (5-15% extra hits)

Architecture:
    Request → cascade_get(exact → hash → semantic)
        → HIT: return cached response (0ms latency, 0 tokens)
        → MISS: call LLM → cascade_put → return
"""

import hashlib
import json
import logging
from typing import Any, Optional

import yaml

from app.config import get_settings

logger = logging.getLogger(__name__)


# ─── Config Loader ──────────────────────────────────────────

def _load_cache_config() -> dict:
    """Load cache config from intelligence_config.yaml."""
    from pathlib import Path
    config_path = Path("intelligence_config.yaml")
    if not config_path.exists():
        return {}
    with open(config_path) as f:
        config = yaml.safe_load(f) or {}
    return config.get("response_cache", {})


# ─── Exact Cache (from prompt_cache.py) ─────────────────────

class _ExactCache:
    """
    Redis-based exact prompt cache.

    Key = SHA-256(tenant_id + system_prompt + user_prompt + messages_hash)
    Value = serialized LLM result (JSON)
    """

    KEY_PREFIX = "sutra:cache:prompt:"

    def __init__(self, redis, *, enabled: bool = True, ttl: int = 7200):
        self._redis = redis
        self._enabled = enabled
        self._ttl = ttl

    @classmethod
    def generate_key(cls, tenant_id: int, prompt: str, system_prompt: str = "", messages: list | None = None) -> str:
        """Generate deterministic cache key from all prompt context."""
        payload = {
            "tid": tenant_id,
            "p": prompt,
            "sp": system_prompt,
            "m": messages or []
        }
        dump = json.dumps(payload, sort_keys=True)
        digest = hashlib.sha256(dump.encode()).hexdigest()
        return f"{cls.KEY_PREFIX}{digest}"

    async def get(self, key: str) -> Any | None:
        """Fetch result from cache by key."""
        if not self._enabled or not self._redis:
            return None

        try:
            cached = await self._redis.get(key)
            if cached:
                return json.loads(cached)
        except Exception as e:
            logger.warning(f"ExactCache read error: {e}")
        return None

    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Store result in cache."""
        if not self._enabled or not self._redis:
            return

        try:
            expire = ttl or self._ttl
            await self._redis.setex(key, expire, json.dumps(value))
        except Exception as e:
            logger.warning(f"ExactCache write error: {e}")

    async def flush_all(self) -> int:
        """Flush all prompt cache entries."""
        if not self._redis:
            return 0
        try:
            count = 0
            async for key in self._redis.scan_iter(f"{self.KEY_PREFIX}*"):
                await self._redis.delete(key)
                count += 1
            return count
        except Exception as e:
            logger.warning(f"ExactCache flush error: {e}")
            return 0


# ─── Hash Cache (from response_cache.py) ────────────────────

class _HashCache:
    """
    Redis-backed response cache with content-hash keys.

    Key = SHA-256(agent_type + normalized_prompt + context_hash)
    Value = {content, agent_type, cached, metadata}
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
        """Create a cache key from agent type and prompt."""
        normalized = " ".join(prompt.lower().strip().split())
        raw = f"{agent_type}:{normalized}:{context_hash}"
        content_hash = hashlib.sha256(raw.encode()).hexdigest()[:32]
        return f"cache:response:{content_hash}"

    async def get(self, agent_type: str, prompt: str, context_hash: str = "") -> Optional[dict]:
        """Check cache for an existing response."""
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
                    f"HashCache: HIT for {agent_type} "
                    f"(hits={self._hits}, misses={self._misses})"
                )
                return data

            self._misses += 1
            return None

        except Exception as e:
            logger.warning(f"HashCache: get failed: {e}")
            return None

    async def put(
        self,
        agent_type: str,
        prompt: str,
        response_content: str,
        metadata: dict | None = None,
        context_hash: str = "",
    ) -> bool:
        """Cache a response for future identical queries."""
        config = _load_cache_config()
        if not config.get("enabled", True):
            return False

        min_length = config.get("min_response_length", 50)
        if len(response_content) < min_length:
            return False

        try:
            redis = await self._get_redis()
            key = self._make_key(agent_type, prompt, context_hash)

            agent_ttls = config.get("agent_ttl", {})
            ttl = agent_ttls.get(agent_type, config.get("default_ttl", 3600))

            data = {
                "content": response_content,
                "agent_type": agent_type,
                "cached": True,
                "metadata": metadata or {},
            }

            await redis.set(key, json.dumps(data, ensure_ascii=False), ex=ttl)
            logger.debug(f"HashCache: stored for {agent_type} (ttl={ttl}s)")
            return True

        except Exception as e:
            logger.warning(f"HashCache: put failed: {e}")
            return False

    async def invalidate(self, agent_type: str, prompt: str, context_hash: str = "") -> bool:
        """Invalidate a specific cached response."""
        try:
            redis = await self._get_redis()
            key = self._make_key(agent_type, prompt, context_hash)
            await redis.delete(key)
            return True
        except Exception as e:
            logger.warning(f"HashCache: invalidate failed: {e}")
            return False

    async def clear_agent(self, agent_type: str) -> int:
        """Clear all cached responses for a specific agent."""
        try:
            redis = await self._get_redis()
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
            logger.info(f"HashCache: cleared {deleted} entries for {agent_type}")
            return deleted
        except Exception as e:
            logger.warning(f"HashCache: clear failed: {e}")
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


# ─── Semantic Cache (from semantic_cache.py) ────────────────

class _SemanticCache:
    """
    ChromaDB-based semantic similarity cache.

    1. Embed the user prompt
    2. Query ChromaDB for similar prompts (cosine similarity > threshold)
    3. If hit → return cached response
    4. If miss → after LLM call, store prompt + response for future hits
    """

    COLLECTION_NAME = "sutra_prompt_cache"

    def __init__(self, chromadb_client, *, enabled: bool = True, similarity_threshold: float = 0.92):
        self._client = chromadb_client
        self._enabled = enabled
        self._threshold = similarity_threshold
        self._collection = None

    async def _get_collection(self):
        """Lazy-init the ChromaDB collection."""
        if self._collection is None:
            self._collection = self._client.get_or_create_collection(
                name=self.COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def _make_id(self, agent_type: str, prompt: str) -> str:
        """Generate a unique ID for a prompt."""
        payload = f"{agent_type}:{prompt}"
        return hashlib.md5(payload.encode()).hexdigest()

    async def get(self, agent_type: str, prompt: str) -> Optional[dict]:
        """Query for a semantically similar cached prompt."""
        if not self._enabled:
            return None

        try:
            collection = await self._get_collection()
            results = collection.query(
                query_texts=[prompt],
                n_results=1,
                where={"agent_type": agent_type},
            )

            if (
                results
                and results["distances"]
                and results["distances"][0]
                and results["distances"][0][0] <= (1 - self._threshold)
            ):
                metadata = results["metadatas"][0][0]
                cached_response = json.loads(metadata.get("response", "{}"))
                logger.info(
                    f"SemanticCache HIT: similarity={1 - results['distances'][0][0]:.3f}, "
                    f"agent={agent_type}"
                )
                return cached_response

        except Exception as e:
            logger.warning(f"SemanticCache query error: {e}")

        return None

    async def set(self, agent_type: str, prompt: str, response: dict) -> None:
        """Store a prompt + response for future similarity matching."""
        if not self._enabled:
            return

        try:
            collection = await self._get_collection()
            doc_id = self._make_id(agent_type, prompt)

            collection.upsert(
                ids=[doc_id],
                documents=[prompt],
                metadatas=[{
                    "agent_type": agent_type,
                    "response": json.dumps(response),
                }],
            )
            logger.debug(f"SemanticCache SET: agent={agent_type}, id={doc_id[:16]}...")

        except Exception as e:
            logger.warning(f"SemanticCache store error: {e}")


# ─── Unified Cache Engine ───────────────────────────────────

class CacheEngine:
    """
    Unified cache with strategy pattern.

    Strategies:
        exact:    tenant-scoped, full context hash (PromptCache)
        hash:     agent+prompt hash with per-agent TTL (ResponseCache)
        semantic: ChromaDB cosine similarity (SemanticCache)

    cascade_get() tries exact → hash → semantic in order. First hit wins.
    """

    def __init__(
        self,
        *,
        exact_cache: _ExactCache | None = None,
        hash_cache: _HashCache | None = None,
        semantic_cache: _SemanticCache | None = None,
    ):
        self.exact = exact_cache
        self.hash = hash_cache or _HashCache()
        self.semantic = semantic_cache

    async def cascade_get(self, agent_type: str, prompt: str, **kwargs) -> Optional[dict]:
        """Try hash → semantic in order. First hit wins."""
        # Hash cache (most common hit)
        result = await self.hash.get(agent_type, prompt, kwargs.get("context_hash", ""))
        if result:
            return result

        # Semantic cache (catches paraphrased queries)
        if self.semantic:
            result = await self.semantic.get(agent_type, prompt)
            if result:
                return result

        return None

    async def cascade_put(
        self,
        agent_type: str,
        prompt: str,
        response_content: str,
        metadata: dict | None = None,
        **kwargs,
    ) -> None:
        """Store in both hash and semantic caches."""
        # Hash cache
        await self.hash.put(
            agent_type, prompt, response_content, metadata, kwargs.get("context_hash", "")
        )

        # Semantic cache
        if self.semantic:
            await self.semantic.set(agent_type, prompt, {
                "content": response_content,
                "agent_type": agent_type,
                "metadata": metadata or {},
            })

    def stats(self) -> dict:
        """Aggregate stats from all cache layers."""
        return {
            "hash": self.hash.stats(),
        }


# ─── Singletons (Backward Compatible) ──────────────────────

_exact_cache: _ExactCache | None = None
_hash_cache: _HashCache | None = None
_engine: CacheEngine | None = None


def get_prompt_cache() -> _ExactCache:
    """Backward-compatible accessor for PromptCache (now _ExactCache)."""
    global _exact_cache
    if _exact_cache is None:
        from app.dependencies import get_redis_sync
        settings = get_settings()
        _exact_cache = _ExactCache(
            get_redis_sync(),
            enabled=settings.ai_prompt_cache_enabled,
            ttl=settings.ai_prompt_cache_ttl,
        )
    return _exact_cache


def get_response_cache() -> _HashCache:
    """Backward-compatible accessor for ResponseCache (now _HashCache)."""
    global _hash_cache
    if _hash_cache is None:
        _hash_cache = _HashCache()
    return _hash_cache


def get_cache_engine() -> CacheEngine:
    """Get the global CacheEngine singleton."""
    global _engine
    if _engine is None:
        _engine = CacheEngine(
            exact_cache=get_prompt_cache(),
            hash_cache=get_response_cache(),
        )
    return _engine
