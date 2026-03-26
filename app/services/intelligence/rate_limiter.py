"""
Rate Limiter — per-tenant request throttling via Redis sliding window.

Stability impact: prevents any single tenant from overwhelming the service.
Uses a sliding window counter so limits don't reset at arbitrary boundaries.
"""

import logging
import time

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Sliding window rate limiter backed by Redis sorted sets.

    Default: 60 requests per minute per tenant.
    Configurable per-tenant via tenant.rate_limits config.
    """

    KEY_PREFIX = "sutra:ratelimit"

    def __init__(self, redis, *, default_rpm: int = 60, default_rpd: int = 10000):
        self._redis = redis
        self._default_rpm = default_rpm
        self._default_rpd = default_rpd

    async def check(self, tenant_id: int, tenant_config: dict | None = None) -> dict:
        """
        Check if a request is allowed.

        Returns: {allowed: bool, remaining: int, limit: int, retry_after: float | None}
        """
        config = tenant_config or {}
        rpm_limit = config.get("rpm", self._default_rpm)

        key = f"{self.KEY_PREFIX}:{tenant_id}:rpm"
        now = time.time()
        window = 60  # 1 minute sliding window

        try:
            pipe = self._redis.pipeline()
            # Remove entries older than the window
            pipe.zremrangebyscore(key, 0, now - window)
            # Count entries in current window
            pipe.zcard(key)
            # Add current request
            pipe.zadd(key, {f"{now}": now})
            # Set expiry on the key
            pipe.expire(key, window + 1)
            results = await pipe.execute()

            current_count = results[1]
            allowed = current_count < rpm_limit
            remaining = max(0, rpm_limit - current_count - 1)

            if not allowed:
                # Find when the next slot opens
                oldest_entries = await self._redis.zrange(key, 0, 0, withscores=True)
                if oldest_entries:
                    retry_after = oldest_entries[0][1] + window - now
                else:
                    retry_after = 1.0

                logger.warning(f"RateLimiter: tenant={tenant_id} rate limited ({current_count}/{rpm_limit} rpm)")
                return {
                    "allowed": False,
                    "remaining": 0,
                    "limit": rpm_limit,
                    "retry_after": round(retry_after, 1),
                }

            return {
                "allowed": True,
                "remaining": remaining,
                "limit": rpm_limit,
                "retry_after": None,
            }

        except Exception as e:
            # On Redis failure, allow the request (fail open)
            logger.warning(f"RateLimiter: Redis error, allowing request: {e}")
            return {"allowed": True, "remaining": -1, "limit": rpm_limit, "retry_after": None}
