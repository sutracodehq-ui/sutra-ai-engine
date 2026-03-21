"""
Rate Limiter — Enforces tier-based API call limits.

Software Factory Principle: Fair usage, no abuse, graceful degradation.

Tiers:
  Free:       50 calls/day,  5 agents
  Starter:   500 calls/day, 25 agents
  Pro:      2000 calls/day, all agents + voice
  Enterprise: unlimited,    all + custom + priority
"""

import logging
from dataclasses import dataclass

import yaml
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class TierLimits:
    """Rate limits for a subscription tier."""
    name: str
    daily_limit: int            # Max API calls per day (0 = unlimited)
    allowed_agents: list[str]   # Agent IDs allowed (empty = all)
    voice_enabled: bool = False
    websocket_enabled: bool = False
    priority_queue: bool = False
    custom_agents: bool = False
    max_concurrent: int = 5     # Max simultaneous requests


# ─── Default Tier Definitions ───────────────────────────────

TIERS: dict[str, TierLimits] = {
    "free": TierLimits(
        name="Free",
        daily_limit=50,
        allowed_agents=[
            "copywriter", "seo", "social", "email_summarizer", "daily_briefing",
        ],
        max_concurrent=2,
    ),
    "starter": TierLimits(
        name="Starter",
        daily_limit=500,
        allowed_agents=[
            # Core marketing
            "copywriter", "seo", "social", "email_campaign", "whatsapp", "sms",
            "ad_creative", "brand_auditor", "content_repurposer",
            # Intelligence
            "persona_builder", "campaign_strategist", "competitor_analyst",
            # Analytics
            "performance_reporter", "budget_optimizer",
            # Creative
            "visual_designer", "landing_page_builder",
            # Productivity
            "email_summarizer", "meeting_notes", "invoice_generator",
            "expense_tracker", "daily_briefing", "reminder_agent",
        ],
        max_concurrent=5,
    ),
    "pro": TierLimits(
        name="Pro",
        daily_limit=2000,
        allowed_agents=[],  # Empty = ALL agents
        voice_enabled=True,
        websocket_enabled=True,
        max_concurrent=10,
    ),
    "enterprise": TierLimits(
        name="Enterprise",
        daily_limit=0,  # 0 = unlimited
        allowed_agents=[],  # All agents
        voice_enabled=True,
        websocket_enabled=True,
        priority_queue=True,
        custom_agents=True,
        max_concurrent=50,
    ),
}


class RateLimiter:
    """
    Enforces tier-based rate limits.
    
    Checks:
    1. Daily call limit
    2. Agent access (is this agent in their tier?)
    3. Feature access (voice, websocket)
    """

    def __init__(self):
        self._tiers = TIERS

    def check(
        self,
        tenant_id: str,
        tier: str,
        agent_id: str,
        current_daily_usage: int,
    ) -> dict:
        """
        Check if a request is allowed.
        
        Returns:
            {"allowed": True} or
            {"allowed": False, "reason": "...", "limit": N, "used": N}
        """
        limits = self._tiers.get(tier)
        if not limits:
            return {"allowed": False, "reason": f"Unknown tier: {tier}"}

        # 1. Check daily limit
        if limits.daily_limit > 0 and current_daily_usage >= limits.daily_limit:
            return {
                "allowed": False,
                "reason": "daily_limit_exceeded",
                "message": f"Daily limit of {limits.daily_limit} calls reached. "
                           f"Upgrade to {self._suggest_upgrade(tier)} for more.",
                "limit": limits.daily_limit,
                "used": current_daily_usage,
                "upgrade_to": self._suggest_upgrade(tier),
            }

        # 2. Check agent access
        if limits.allowed_agents and agent_id not in limits.allowed_agents:
            return {
                "allowed": False,
                "reason": "agent_not_in_tier",
                "message": f"Agent '{agent_id}' is not available on the {limits.name} plan. "
                           f"Upgrade to {self._suggest_upgrade(tier)} to access it.",
                "upgrade_to": self._suggest_upgrade(tier),
            }

        # All checks passed
        return {
            "allowed": True,
            "remaining": (limits.daily_limit - current_daily_usage - 1)
                         if limits.daily_limit > 0 else -1,  # -1 = unlimited
            "tier": tier,
        }

    def check_feature(self, tier: str, feature: str) -> bool:
        """Check if a feature is available for a tier."""
        limits = self._tiers.get(tier)
        if not limits:
            return False

        feature_map = {
            "voice": limits.voice_enabled,
            "websocket": limits.websocket_enabled,
            "priority": limits.priority_queue,
            "custom_agents": limits.custom_agents,
        }
        return feature_map.get(feature, False)

    def get_tier_info(self, tier: str) -> dict | None:
        """Get tier details for display."""
        limits = self._tiers.get(tier)
        if not limits:
            return None

        return {
            "name": limits.name,
            "daily_limit": limits.daily_limit if limits.daily_limit > 0 else "unlimited",
            "agents": "all" if not limits.allowed_agents else len(limits.allowed_agents),
            "voice": limits.voice_enabled,
            "websocket": limits.websocket_enabled,
            "priority": limits.priority_queue,
            "custom_agents": limits.custom_agents,
            "max_concurrent": limits.max_concurrent,
        }

    def all_tiers(self) -> dict:
        """Get all tier info for pricing page."""
        return {k: self.get_tier_info(k) for k in self._tiers}

    @staticmethod
    def _suggest_upgrade(current_tier: str) -> str:
        upgrades = {"free": "Starter", "starter": "Pro", "pro": "Enterprise"}
        return upgrades.get(current_tier, "Enterprise")


# ─── Singleton ──────────────────────────────────────────────

_limiter: RateLimiter | None = None


def get_rate_limiter() -> RateLimiter:
    global _limiter
    if _limiter is None:
        _limiter = RateLimiter()
    return _limiter
