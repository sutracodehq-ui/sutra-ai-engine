"""
Secure Gateway — Wraps billing gateway with full security pipeline.

Full request flow:
    Request → IP Check → Anti-Replay → HMAC Verify → Injection Guard →
    PII Redact → Rate Limit → Cache Check → Agent Execute →
    Track Usage → Audit Log → Return Response

This is THE single entry point for all external API calls.
"""

import logging
import time
from typing import Any

from app.services.billing.api_keys import get_api_key_manager
from app.services.billing.usage_tracker import get_usage_tracker
from app.services.billing.rate_limiter import get_rate_limiter
from app.services.security.injection_guard import get_injection_guard
from app.services.security.pii_redactor import get_pii_redactor
from app.services.security.request_auth import get_request_authenticator
from app.services.security.audit_logger import get_audit_logger
from app.services.optimization.response_cache import get_response_cache

logger = logging.getLogger(__name__)


class SecureGateway:
    """
    Production-grade entry point with 8-layer security pipeline.
    
    Layers:
    1. API Key authentication
    2. IP whitelist check 
    3. Anti-replay (nonce + timestamp)
    4. HMAC signature verification
    5. Prompt injection detection
    6. PII redaction
    7. Rate limiting
    8. Response caching
    """

    def __init__(self):
        self._keys = get_api_key_manager()
        self._tracker = get_usage_tracker()
        self._limiter = get_rate_limiter()
        self._guard = get_injection_guard()
        self._pii = get_pii_redactor()
        self._auth = get_request_authenticator()
        self._audit = get_audit_logger()
        self._cache = get_response_cache()

    async def execute(
        self,
        api_key: str,
        agent_id: str,
        prompt: str,
        context: dict | None = None,
        db: Any | None = None,
        request_ip: str = "",
        nonce: str = "",
        timestamp: str = "",
        signature: str = "",
    ) -> dict:
        """Full secure + metered agent execution."""
        start = time.time()
        tenant_id = ""
        tier = ""

        try:
            # ── 1. Authenticate API Key ──
            key_info = self._keys.validate(api_key)
            if not key_info:
                self._audit.log_security_event("unknown", "invalid_api_key", request_ip)
                return self._error("invalid_api_key", "Invalid or expired API key.", 401)

            tenant_id = key_info.tenant_id
            tier = key_info.tier

            # ── 2. IP + Anti-Replay + HMAC ──
            auth_result = self._auth.full_check(
                tenant_id=tenant_id,
                request_ip=request_ip,
                payload=prompt,
                timestamp=timestamp,
                signature=signature,
                nonce=nonce,
            )
            if not auth_result["passed"]:
                self._audit.log_security_event(tenant_id, f"auth_failed:{auth_result['check']}", request_ip, "critical")
                return self._error(auth_result.get("reason", "auth_failed"), auth_result.get("message", "Authentication failed."), 403)

            # ── 3. Prompt Injection Guard ──
            injection = self._guard.check(prompt)
            if not injection.is_safe:
                self._audit.log_security_event(tenant_id, f"injection_blocked:{injection.triggers}", request_ip, "critical")
                return self._error("prompt_injection_detected",
                    "Your request was blocked for security reasons. Please rephrase.", 400)

            # ── 4. PII Redaction ──
            pii_result = self._pii.redact(prompt)
            safe_prompt = pii_result.redacted

            # ── 5. Rate Limit ──
            daily_usage = self._tracker.get_daily_total(tenant_id)
            limit_check = self._limiter.check(tenant_id, tier, agent_id, daily_usage)
            if not limit_check["allowed"]:
                self._audit.log_request(
                    tenant_id=tenant_id, agent_id=agent_id, prompt=safe_prompt,
                    ip_address=request_ip, tier=tier, rate_limited=True, success=False,
                )
                return self._error(limit_check["reason"], limit_check.get("message", "Rate limit exceeded."), 429,
                    extra={"limit": limit_check.get("limit"), "used": limit_check.get("used"), "upgrade_to": limit_check.get("upgrade_to")})

            # ── 6. Cache Check ──
            cached = self._cache.get(tenant_id, agent_id, safe_prompt)
            if cached:
                latency = int((time.time() - start) * 1000)
                self._audit.log_request(
                    tenant_id=tenant_id, agent_id=agent_id, prompt=safe_prompt,
                    response=cached, ip_address=request_ip, tier=tier,
                    latency_ms=latency, pii_detected=pii_result.pii_count,
                )
                return {
                    "success": True, "response": cached, "agent": agent_id,
                    "cached": True,
                    "usage": {"remaining": limit_check.get("remaining", -1), "tier": tier, "latency_ms": latency},
                }

            # ── 7. Execute Agent ──
            from app.services.agents.hub import get_agent_hub
            hub = get_agent_hub()

            result = await hub.run(
                agent_type=agent_id, prompt=safe_prompt, db=db,
                context={**(context or {}), "_tenant_id": tenant_id, "_tier": tier},
            )

            response_text = result.content if hasattr(result, "content") else str(result)
            latency = int((time.time() - start) * 1000)

            # ── 8. Cache + Track + Audit ──
            self._cache.set(tenant_id, agent_id, safe_prompt, response_text)

            if not key_info.is_test:
                self._tracker.track(
                    tenant_id=tenant_id, agent_id=agent_id,
                    tokens_used=getattr(result, "tokens_used", 0),
                    latency_ms=latency, success=True,
                )

            self._audit.log_request(
                tenant_id=tenant_id, agent_id=agent_id, prompt=safe_prompt,
                response=response_text, ip_address=request_ip, tier=tier,
                latency_ms=latency, pii_detected=pii_result.pii_count,
                injection_risk=injection.risk_score,
                remaining_calls=limit_check.get("remaining", -1),
            )

            return {
                "success": True, "response": response_text, "agent": agent_id,
                "cached": False,
                "usage": {"remaining": limit_check.get("remaining", -1), "tier": tier, "latency_ms": latency},
                "security": {"pii_redacted": pii_result.pii_count, "injection_risk": round(injection.risk_score, 2)},
            }

        except Exception as e:
            latency = int((time.time() - start) * 1000)
            self._audit.log_request(
                tenant_id=tenant_id, agent_id=agent_id, prompt=prompt,
                ip_address=request_ip, tier=tier, latency_ms=latency,
                success=False, error_message=str(e),
            )
            logger.error(f"SecureGateway: {e}")
            return self._error("internal_error", "An error occurred processing your request.", 500)

    @staticmethod
    def _error(code: str, message: str, status: int, extra: dict | None = None) -> dict:
        return {"success": False, "error": code, "message": message, "status": status, **(extra or {})}


# ─── Singleton ──────────────────────────────────────────────

_gw: SecureGateway | None = None

def get_secure_gateway() -> SecureGateway:
    global _gw
    if _gw is None:
        _gw = SecureGateway()
    return _gw
