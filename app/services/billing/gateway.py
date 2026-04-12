"""
Billing Middleware — Wraps the agent hub with authentication, metering, and rate limiting.

Software Factory Principle: Single gateway for all billable operations.

Flow:
    API Key → Validate → Check Rate Limit → Run Agent → Track Usage → Return Response
"""

import logging
import time
from typing import Any, Optional

from app.services.billing.api_keys import get_api_key_manager, ApiKey
from app.services.billing.usage_tracker import get_usage_tracker
from app.services.billing.rate_limiter import get_rate_limiter

logger = logging.getLogger(__name__)


class BillingGateway:
    """
    Central gateway that authenticates, rate-limits, meters, and dispatches agent calls.
    
    This is the ONLY entry point for external API calls.
    Internal calls (agent-to-agent delegation) bypass billing.
    """

    def __init__(self):
        self._keys = get_api_key_manager()
        self._tracker = get_usage_tracker()
        self._limiter = get_rate_limiter()

    async def execute(
        self,
        api_key: str,
        agent_id: str,
        prompt: str,
        context: dict | None = None,
        db: Any | None = None,
        **options,
    ) -> dict:
        """
        Authenticated + metered agent execution.
        
        Returns:
            {
                "success": True,
                "response": "...",
                "usage": {"remaining": 42, "tier": "starter"},
            }
            OR
            {
                "success": False,
                "error": "rate_limit_exceeded",
                "message": "...",
            }
        """
        start_time = time.time()

        # ── 1. Authenticate ──
        key_info = self._keys.validate(api_key)
        if not key_info:
            return {
                "success": False,
                "error": "invalid_api_key",
                "message": "Invalid or expired API key.",
            }

        tenant_id = key_info.tenant_id
        tier = key_info.tier

        # ── 2. Rate limit check ──
        daily_usage = self._tracker.get_daily_total(tenant_id)
        limit_check = self._limiter.check(
            tenant_id=tenant_id,
            tier=tier,
            agent_id=agent_id,
            current_daily_usage=daily_usage,
        )

        if not limit_check["allowed"]:
            return {
                "success": False,
                "error": limit_check["reason"],
                "message": limit_check.get("message", "Rate limit exceeded."),
                "limit": limit_check.get("limit"),
                "used": limit_check.get("used"),
                "upgrade_to": limit_check.get("upgrade_to"),
            }

        # ── 3. Execute agent ──
        from app.services.agents.hub import get_agent_hub
        hub = get_agent_hub()

        try:
            result = await hub.run(
                agent_type=agent_id,
                prompt=prompt,
                db=db,
                context={
                    **(context or {}),
                    "_tenant_id": tenant_id,
                    "_tier": tier,
                },
                **options,
            )

            latency_ms = int((time.time() - start_time) * 1000)

            # ── 4. Track usage (only on success) ──
            if not key_info.is_test:  # Don't count test mode
                self._tracker.track(
                    tenant_id=tenant_id,
                    agent_id=agent_id,
                    tokens_used=getattr(result, "tokens_used", 0),
                    latency_ms=latency_ms,
                    success=True,
                )

            return {
                "success": True,
                "response": result.content if hasattr(result, "content") else str(result),
                "agent": agent_id,
                "usage": {
                    "remaining": limit_check.get("remaining", -1),
                    "tier": tier,
                    "latency_ms": latency_ms,
                },
            }

        except ValueError as e:
            # Agent not found
            return {
                "success": False,
                "error": "agent_not_found",
                "message": str(e),
            }
        except Exception as e:
            latency_ms = int((time.time() - start_time) * 1000)

            # Track failed calls too
            if not key_info.is_test:
                self._tracker.track(
                    tenant_id=tenant_id,
                    agent_id=agent_id,
                    latency_ms=latency_ms,
                    success=False,
                )

            logger.error(f"BillingGateway: agent '{agent_id}' failed for tenant '{tenant_id}': {e}")
            return {
                "success": False,
                "error": "agent_error",
                "message": "An error occurred processing your request.",
            }


# ─── Singleton ──────────────────────────────────────────────

_gateway: BillingGateway | None = None


def get_billing_gateway() -> BillingGateway:
    global _gateway
    if _gateway is None:
        _gateway = BillingGateway()
    return _gateway
