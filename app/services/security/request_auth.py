"""
Request Authenticator — HMAC signing, IP whitelist, anti-replay.

Security Layer: Ensures request integrity, source verification, and replay protection.

Features:
- HMAC-SHA256 request signing (tamper-proof)
- Per-tenant IP whitelisting
- Nonce + timestamp anti-replay (5-minute window)
- Request fingerprinting
"""

import hashlib
import hmac
import logging
import time
from collections import defaultdict

logger = logging.getLogger(__name__)


class RequestAuthenticator:
    """
    Multi-layer request authentication.
    
    Flow:
        1. Check IP whitelist (if configured)
        2. Verify HMAC signature (if signing enabled)
        3. Anti-replay check (nonce + timestamp)
    """

    REPLAY_WINDOW = 300  # 5 minutes

    def __init__(self):
        # Per-tenant IP whitelists: {tenant_id: set(ips)}
        self._ip_whitelist: dict[str, set[str]] = {}
        # Per-tenant signing secrets: {tenant_id: secret}
        self._signing_secrets: dict[str, str] = {}
        # Nonce cache: {nonce: expiry_time}
        self._nonce_cache: dict[str, float] = {}

    # ─── IP Whitelisting ──────────────────────────────────

    def set_ip_whitelist(self, tenant_id: str, allowed_ips: list[str]):
        """Configure allowed IPs for a tenant."""
        self._ip_whitelist[tenant_id] = set(allowed_ips)
        logger.info(f"IP whitelist set for '{tenant_id}': {allowed_ips}")

    def check_ip(self, tenant_id: str, request_ip: str) -> dict:
        """Check if request IP is allowed."""
        allowed = self._ip_whitelist.get(tenant_id)
        if allowed is None:
            return {"allowed": True, "reason": "no_whitelist_configured"}

        if request_ip in allowed:
            return {"allowed": True}

        logger.warning(f"IP blocked: {request_ip} not in whitelist for '{tenant_id}'")
        return {
            "allowed": False,
            "reason": "ip_not_whitelisted",
            "message": f"IP {request_ip} is not authorized for this API key.",
        }

    # ─── HMAC Request Signing ─────────────────────────────

    def set_signing_secret(self, tenant_id: str, secret: str):
        """Set the HMAC signing secret for a tenant."""
        self._signing_secrets[tenant_id] = secret

    def generate_signature(self, tenant_id: str, payload: str, timestamp: str) -> str | None:
        """Generate HMAC-SHA256 signature for a payload."""
        secret = self._signing_secrets.get(tenant_id)
        if not secret:
            return None

        message = f"{timestamp}.{payload}"
        signature = hmac.new(
            secret.encode(), message.encode(), hashlib.sha256
        ).hexdigest()
        return f"v1={signature}"

    def verify_signature(
        self,
        tenant_id: str,
        payload: str,
        timestamp: str,
        signature: str,
    ) -> dict:
        """Verify HMAC-SHA256 signature."""
        secret = self._signing_secrets.get(tenant_id)
        if not secret:
            return {"valid": True, "reason": "signing_not_configured"}

        # Check timestamp freshness
        try:
            ts = int(timestamp)
            if abs(time.time() - ts) > self.REPLAY_WINDOW:
                return {
                    "valid": False,
                    "reason": "timestamp_expired",
                    "message": "Request timestamp is too old (>5 minutes).",
                }
        except ValueError:
            return {"valid": False, "reason": "invalid_timestamp"}

        expected = self.generate_signature(tenant_id, payload, timestamp)
        if not hmac.compare_digest(signature, expected):
            logger.warning(f"HMAC verification failed for tenant '{tenant_id}'")
            return {
                "valid": False,
                "reason": "invalid_signature",
                "message": "Request signature verification failed.",
            }

        return {"valid": True}

    # ─── Anti-Replay ──────────────────────────────────────

    def check_replay(self, nonce: str, timestamp: str) -> dict:
        """
        Prevent replay attacks with nonce + timestamp.
        Each nonce can only be used once within the replay window.
        """
        # Clean old nonces
        now = time.time()
        expired_nonces = [n for n, exp in self._nonce_cache.items() if now > exp]
        for n in expired_nonces:
            del self._nonce_cache[n]

        # Check timestamp freshness
        try:
            ts = int(timestamp)
            if abs(now - ts) > self.REPLAY_WINDOW:
                return {
                    "valid": False,
                    "reason": "timestamp_expired",
                    "message": "Request is too old.",
                }
        except ValueError:
            return {"valid": False, "reason": "invalid_timestamp"}

        # Check nonce uniqueness
        if nonce in self._nonce_cache:
            logger.warning(f"Replay attack detected: nonce '{nonce}' already used")
            return {
                "valid": False,
                "reason": "nonce_reused",
                "message": "This request has already been processed.",
            }

        # Store nonce with expiry
        self._nonce_cache[nonce] = now + self.REPLAY_WINDOW
        return {"valid": True}

    def full_check(
        self,
        tenant_id: str,
        request_ip: str,
        payload: str = "",
        timestamp: str = "",
        signature: str = "",
        nonce: str = "",
    ) -> dict:
        """
        Run all security checks in sequence.
        Returns on first failure.
        """
        # 1. IP check
        ip_result = self.check_ip(tenant_id, request_ip)
        if not ip_result["allowed"]:
            return {"passed": False, "check": "ip_whitelist", **ip_result}

        # 2. Signature check (if configured)
        if signature:
            sig_result = self.verify_signature(tenant_id, payload, timestamp, signature)
            if not sig_result["valid"]:
                return {"passed": False, "check": "hmac_signature", **sig_result}

        # 3. Anti-replay (if nonce provided)
        if nonce:
            replay_result = self.check_replay(nonce, timestamp)
            if not replay_result["valid"]:
                return {"passed": False, "check": "anti_replay", **replay_result}

        return {"passed": True}


# ─── Singleton ──────────────────────────────────────────────

_auth: RequestAuthenticator | None = None


def get_request_authenticator() -> RequestAuthenticator:
    global _auth
    if _auth is None:
        _auth = RequestAuthenticator()
    return _auth
