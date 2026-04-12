"""
Usage Tracker — Counts API calls per agent per tenant per day.

Software Factory Principle: If you can't measure it, you can't charge for it.

Features:
- Per-agent, per-tenant, per-day usage counters
- Sliding window tracking
- Usage history for billing reports
- Real-time usage queries
"""

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, date

logger = logging.getLogger(__name__)


@dataclass
class UsageRecord:
    """A single usage event."""
    tenant_id: str
    agent_id: str
    timestamp: float
    tokens_used: int = 0
    latency_ms: int = 0
    success: bool = True


class UsageTracker:
    """
    Tracks API usage per tenant.
    
    In production, back this with Redis or a timeseries DB.
    Currently uses in-memory counters for development.
    """

    def __init__(self):
        # Daily counters: {tenant_id: {date_str: {agent_id: count}}}
        self._daily: dict[str, dict[str, dict[str, int]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(int))
        )
        # Total counters: {tenant_id: {agent_id: count}}
        self._totals: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        # Recent records for detailed queries
        self._recent: list[UsageRecord] = []
        self._max_recent = 10000

    def track(
        self,
        tenant_id: str,
        agent_id: str,
        tokens_used: int = 0,
        latency_ms: int = 0,
        success: bool = True,
    ):
        """Record a single API call."""
        today = date.today().isoformat()

        self._daily[tenant_id][today][agent_id] += 1
        self._totals[tenant_id][agent_id] += 1

        record = UsageRecord(
            tenant_id=tenant_id,
            agent_id=agent_id,
            timestamp=time.time(),
            tokens_used=tokens_used,
            latency_ms=latency_ms,
            success=success,
        )
        self._recent.append(record)

        # Trim old records
        if len(self._recent) > self._max_recent:
            self._recent = self._recent[-self._max_recent:]

    def get_daily_usage(self, tenant_id: str, day: str | None = None) -> dict:
        """Get usage for a specific day (default: today)."""
        day = day or date.today().isoformat()
        agents = self._daily.get(tenant_id, {}).get(day, {})
        total = sum(agents.values())
        return {
            "tenant_id": tenant_id,
            "date": day,
            "total_calls": total,
            "by_agent": dict(agents),
        }

    def get_daily_total(self, tenant_id: str, day: str | None = None) -> int:
        """Get total calls today for rate limiting."""
        day = day or date.today().isoformat()
        agents = self._daily.get(tenant_id, {}).get(day, {})
        return sum(agents.values())

    def get_usage_summary(self, tenant_id: str, days: int = 30) -> dict:
        """Get usage summary for last N days."""
        from datetime import timedelta

        today = date.today()
        daily_data = []

        for i in range(days):
            day = (today - timedelta(days=i)).isoformat()
            agents = self._daily.get(tenant_id, {}).get(day, {})
            daily_data.append({
                "date": day,
                "total_calls": sum(agents.values()),
                "by_agent": dict(agents),
            })

        return {
            "tenant_id": tenant_id,
            "period_days": days,
            "total_calls": sum(d["total_calls"] for d in daily_data),
            "daily": daily_data,
            "top_agents": dict(
                sorted(
                    self._totals.get(tenant_id, {}).items(),
                    key=lambda x: x[1],
                    reverse=True,
                )[:10]
            ),
        }

    def get_agent_usage(self, tenant_id: str, agent_id: str) -> dict:
        """Get total usage for a specific agent."""
        total = self._totals.get(tenant_id, {}).get(agent_id, 0)
        return {
            "tenant_id": tenant_id,
            "agent_id": agent_id,
            "total_calls": total,
        }


# ─── Singleton ──────────────────────────────────────────────

_tracker: UsageTracker | None = None


def get_usage_tracker() -> UsageTracker:
    global _tracker
    if _tracker is None:
        _tracker = UsageTracker()
    return _tracker
