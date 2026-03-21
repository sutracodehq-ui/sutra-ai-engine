"""
Audit Logger — Immutable log of every API call for compliance and debugging.

Security Layer: Full traceability of who did what, when, from where.

Logs:
- Tenant, agent, timestamp, IP, latency
- Request/response hashes (not full content — privacy)
- Security events (blocked requests, PII detected, injection attempts)
- Billing events (rate limit hits, tier upgrades)
"""

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class AuditEntry:
    """Single audit log entry."""
    event_id: str                      # Unique event ID
    event_type: str                    # request, security, billing, error
    timestamp: float
    timestamp_iso: str
    tenant_id: str
    agent_id: str = ""
    ip_address: str = ""
    tier: str = ""
    # Request metadata (hashes, not content)
    request_hash: str = ""
    response_hash: str = ""
    prompt_length: int = 0
    response_length: int = 0
    # Performance
    latency_ms: int = 0
    tokens_used: int = 0
    # Security events
    security_flags: list[str] = field(default_factory=list)
    pii_detected: int = 0
    injection_risk: float = 0.0
    # Status
    success: bool = True
    error_message: str = ""
    # Billing
    rate_limited: bool = False
    remaining_calls: int = -1


class AuditLogger:
    """
    Immutable audit log for compliance.
    
    In production: Write to append-only storage (S3, CloudWatch, ELK).
    Currently: In-memory with file rotation for development.
    """

    def __init__(self, log_dir: str = "/tmp/sutracode_audit"):
        import os
        self._log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self._entries: list[AuditEntry] = []
        self._max_memory = 50000  # Keep last 50K in memory

    def log_request(
        self,
        tenant_id: str,
        agent_id: str,
        prompt: str,
        response: str = "",
        ip_address: str = "",
        tier: str = "",
        latency_ms: int = 0,
        tokens_used: int = 0,
        success: bool = True,
        error_message: str = "",
        security_flags: list[str] | None = None,
        pii_detected: int = 0,
        injection_risk: float = 0.0,
        rate_limited: bool = False,
        remaining_calls: int = -1,
    ) -> str:
        """Log an API request with full metadata."""
        import uuid

        event_id = str(uuid.uuid4())[:12]
        now = time.time()

        entry = AuditEntry(
            event_id=event_id,
            event_type="request",
            timestamp=now,
            timestamp_iso=datetime.fromtimestamp(now).isoformat(),
            tenant_id=tenant_id,
            agent_id=agent_id,
            ip_address=ip_address,
            tier=tier,
            request_hash=self._hash(prompt),
            response_hash=self._hash(response),
            prompt_length=len(prompt),
            response_length=len(response),
            latency_ms=latency_ms,
            tokens_used=tokens_used,
            success=success,
            error_message=error_message,
            security_flags=security_flags or [],
            pii_detected=pii_detected,
            injection_risk=injection_risk,
            rate_limited=rate_limited,
            remaining_calls=remaining_calls,
        )

        self._entries.append(entry)
        self._write_to_file(entry)

        # Trim memory
        if len(self._entries) > self._max_memory:
            self._entries = self._entries[-self._max_memory:]

        return event_id

    def log_security_event(
        self,
        tenant_id: str,
        event_detail: str,
        ip_address: str = "",
        severity: str = "warning",
    ) -> str:
        """Log a security-specific event."""
        import uuid

        event_id = str(uuid.uuid4())[:12]
        now = time.time()

        entry = AuditEntry(
            event_id=event_id,
            event_type="security",
            timestamp=now,
            timestamp_iso=datetime.fromtimestamp(now).isoformat(),
            tenant_id=tenant_id,
            ip_address=ip_address,
            security_flags=[f"{severity}: {event_detail}"],
            success=False,
        )

        self._entries.append(entry)
        self._write_to_file(entry)
        logger.warning(f"AUDIT SECURITY: [{tenant_id}] {event_detail}")
        return event_id

    def get_tenant_log(
        self,
        tenant_id: str,
        limit: int = 100,
        event_type: str | None = None,
    ) -> list[dict]:
        """Get audit log for a tenant."""
        filtered = [
            e for e in reversed(self._entries)
            if e.tenant_id == tenant_id
            and (event_type is None or e.event_type == event_type)
        ]
        return [asdict(e) for e in filtered[:limit]]

    def get_security_events(
        self,
        tenant_id: str | None = None,
        limit: int = 50,
    ) -> list[dict]:
        """Get recent security events across all tenants."""
        filtered = [
            e for e in reversed(self._entries)
            if e.event_type == "security"
            and (tenant_id is None or e.tenant_id == tenant_id)
        ]
        return [asdict(e) for e in filtered[:limit]]

    def get_stats(self, tenant_id: str) -> dict:
        """Get audit statistics for a tenant."""
        entries = [e for e in self._entries if e.tenant_id == tenant_id]
        security = [e for e in entries if e.security_flags]
        failed = [e for e in entries if not e.success]

        return {
            "total_requests": len(entries),
            "security_events": len(security),
            "failed_requests": len(failed),
            "avg_latency_ms": (
                sum(e.latency_ms for e in entries) // max(len(entries), 1)
            ),
            "total_pii_detected": sum(e.pii_detected for e in entries),
            "rate_limited_count": sum(1 for e in entries if e.rate_limited),
        }

    def _write_to_file(self, entry: AuditEntry):
        """Append entry to daily log file (append-only)."""
        try:
            date_str = datetime.fromtimestamp(entry.timestamp).strftime("%Y-%m-%d")
            filepath = f"{self._log_dir}/audit_{date_str}.jsonl"
            with open(filepath, "a") as f:
                f.write(json.dumps(asdict(entry)) + "\n")
        except Exception as e:
            logger.error(f"AuditLogger: failed to write to file: {e}")

    @staticmethod
    def _hash(content: str) -> str:
        """Hash content for audit (privacy-safe)."""
        if not content:
            return ""
        return hashlib.sha256(content.encode()).hexdigest()[:16]


# ─── Singleton ──────────────────────────────────────────────

_audit: AuditLogger | None = None


def get_audit_logger() -> AuditLogger:
    global _audit
    if _audit is None:
        _audit = AuditLogger()
    return _audit
