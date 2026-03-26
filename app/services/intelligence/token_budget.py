"""
Token Budget Manager — per-tenant cost control and usage tracking.

Stability impact: prevents runaway token consumption. Enforces monthly budgets
with soft and hard limits (warn at 80%, block at 100%).
"""

import json
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class TokenBudgetManager:
    """
    Per-tenant token budget enforcement via Redis.

    Redis keys:
    - sutra:budget:{tenant_id}:monthly:{YYYY-MM}:tokens → total tokens used
    - sutra:budget:{tenant_id}:monthly:{YYYY-MM}:cost → total cost in USD

    Enforcement levels:
    - ALLOW: under 80% of budget → normal operation
    - WARN: 80-100% of budget → log warning, allow but flag
    - BLOCK: over 100% of budget → reject the request
    """

    KEY_PREFIX = "sutra:budget"

    # Cost per 1K tokens (approximate, varies by model)
    DEFAULT_COSTS = {
        "gpt-4o": {"input": 0.0025, "output": 0.010},
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
        "claude-sonnet-4-20250514": {"input": 0.003, "output": 0.015},
        "claude-haiku-4-20250514": {"input": 0.0008, "output": 0.004},
        "gemini-2.5-pro-preview-06-05": {"input": 0.00125, "output": 0.010},
        "gemini-2.0-flash": {"input": 0.0001, "output": 0.0004},
        "llama-3.3-70b-versatile": {"input": 0.00059, "output": 0.00079},
        "mock": {"input": 0.0, "output": 0.0},
    }

    def __init__(self, redis, *, default_monthly_limit: int = 1_000_000):
        self._redis = redis
        self._default_monthly_limit = default_monthly_limit

    def _period_key(self, tenant_id: int) -> str:
        """Current month key."""
        month = datetime.now(timezone.utc).strftime("%Y-%m")
        return f"{self.KEY_PREFIX}:{tenant_id}:monthly:{month}"

    async def check_budget(self, tenant_id: int, tenant_config: dict | None = None) -> dict:
        """
        Check if a tenant has budget remaining.

        Returns: {allowed: bool, usage: int, limit: int, percentage: float, level: str}
        """
        monthly_limit = (tenant_config or {}).get("monthly_token_limit", self._default_monthly_limit)
        key = self._period_key(tenant_id)

        try:
            usage = int(await self._redis.get(f"{key}:tokens") or 0)
        except Exception:
            usage = 0

        percentage = (usage / monthly_limit * 100) if monthly_limit > 0 else 0

        if percentage >= 100:
            level = "BLOCK"
            allowed = False
        elif percentage >= 80:
            level = "WARN"
            allowed = True
            logger.warning(f"TokenBudget: tenant={tenant_id} at {percentage:.0f}% ({usage}/{monthly_limit})")
        else:
            level = "ALLOW"
            allowed = True

        return {
            "allowed": allowed,
            "usage": usage,
            "limit": monthly_limit,
            "percentage": round(percentage, 1),
            "level": level,
        }

    async def record_usage(
        self,
        tenant_id: int,
        tokens: int,
        model: str = "unknown",
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ) -> None:
        """Record token usage after a successful call."""
        key = self._period_key(tenant_id)

        # Calculate cost
        costs = self.DEFAULT_COSTS.get(model, {"input": 0.001, "output": 0.002})
        cost = (prompt_tokens / 1000 * costs["input"]) + (completion_tokens / 1000 * costs["output"])

        try:
            pipe = self._redis.pipeline()
            pipe.incrby(f"{key}:tokens", tokens)
            pipe.incrbyfloat(f"{key}:cost", cost)
            # Set expiry to 60 days (keep 2 months of data)
            pipe.expire(f"{key}:tokens", 60 * 86400)
            pipe.expire(f"{key}:cost", 60 * 86400)
            await pipe.execute()

            logger.debug(f"TokenBudget: tenant={tenant_id} +{tokens} tokens, +${cost:.6f}")
        except Exception as e:
            logger.warning(f"TokenBudget: failed to record usage: {e}")

    async def get_usage(self, tenant_id: int) -> dict:
        """Get current period usage for a tenant."""
        key = self._period_key(tenant_id)
        try:
            tokens = int(await self._redis.get(f"{key}:tokens") or 0)
            cost = float(await self._redis.get(f"{key}:cost") or 0)
            return {
                "period": datetime.now(timezone.utc).strftime("%Y-%m"),
                "total_tokens": tokens,
                "total_cost_usd": round(cost, 6),
            }
        except Exception:
            return {"period": "", "total_tokens": 0, "total_cost_usd": 0.0}
