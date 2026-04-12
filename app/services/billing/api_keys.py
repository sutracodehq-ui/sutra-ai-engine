"""
API Key Manager — Generate, validate, and revoke API keys per tenant.

Software Factory Principle: Every API call is metered.

Features:
- Cryptographically secure key generation (sc_live_xxx, sc_test_xxx)
- Key → tenant mapping with tier info
- Key rotation without downtime
- Test mode keys (free, no billing)
"""

import hashlib
import logging
import secrets
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ApiKey:
    """Represents a tenant's API key."""
    key_id: str                     # Short ID (first 8 chars)
    key_hash: str                   # SHA-256 hash (stored, never the raw key)
    tenant_id: str                  # Owner tenant/organization
    tier: str = "free"              # free, starter, pro, enterprise
    name: str = ""                  # Human-readable label
    is_test: bool = False           # Test mode (no billing)
    is_active: bool = True
    created_at: float = field(default_factory=time.time)
    last_used_at: Optional[float] = None
    expires_at: Optional[float] = None  # None = never expires


class ApiKeyManager:
    """
    Manages API keys for tenants.
    
    In production, back this with a database table.
    Currently uses in-memory store for development.
    """

    PREFIX_LIVE = "sc_live_"
    PREFIX_TEST = "sc_test_"

    def __init__(self):
        self._keys: dict[str, ApiKey] = {}      # key_hash → ApiKey
        self._key_index: dict[str, str] = {}    # key_id → key_hash (for lookup)
        self._tenant_keys: dict[str, list[str]] = {}  # tenant_id → [key_hashes]

    def generate(
        self,
        tenant_id: str,
        tier: str = "free",
        name: str = "",
        is_test: bool = False,
    ) -> str:
        """
        Generate a new API key for a tenant.
        Returns the raw key (only shown once, never stored).
        """
        prefix = self.PREFIX_TEST if is_test else self.PREFIX_LIVE
        raw_key = prefix + secrets.token_urlsafe(32)
        key_hash = self._hash(raw_key)
        key_id = raw_key[:len(prefix) + 8]

        api_key = ApiKey(
            key_id=key_id,
            key_hash=key_hash,
            tenant_id=tenant_id,
            tier=tier,
            name=name or f"{tenant_id}-{'test' if is_test else 'live'}",
            is_test=is_test,
        )

        self._keys[key_hash] = api_key
        self._key_index[key_id] = key_hash
        self._tenant_keys.setdefault(tenant_id, []).append(key_hash)

        logger.info(f"ApiKeyManager: generated key {key_id} for tenant '{tenant_id}' (tier={tier})")
        return raw_key

    def validate(self, raw_key: str) -> Optional[ApiKey]:
        """
        Validate an API key.
        Returns ApiKey if valid, None if invalid/expired/revoked.
        """
        key_hash = self._hash(raw_key)
        api_key = self._keys.get(key_hash)

        if not api_key:
            return None

        if not api_key.is_active:
            return None

        if api_key.expires_at and time.time() > api_key.expires_at:
            return None

        # Update last used
        api_key.last_used_at = time.time()
        return api_key

    def revoke(self, key_id: str) -> bool:
        """Revoke an API key by its ID prefix."""
        key_hash = self._key_index.get(key_id)
        if not key_hash or key_hash not in self._keys:
            return False

        self._keys[key_hash].is_active = False
        logger.info(f"ApiKeyManager: revoked key {key_id}")
        return True

    def rotate(self, tenant_id: str, old_key_id: str) -> Optional[str]:
        """Rotate a key: generate new one with same tier, then revoke old."""
        old_hash = self._key_index.get(old_key_id)
        if not old_hash or old_hash not in self._keys:
            return None

        old_key = self._keys[old_hash]
        new_raw = self.generate(
            tenant_id=tenant_id,
            tier=old_key.tier,
            name=old_key.name + " (rotated)",
            is_test=old_key.is_test,
        )
        self.revoke(old_key_id)
        return new_raw

    def list_keys(self, tenant_id: str) -> list[dict]:
        """List all keys for a tenant (safe info only, no hashes)."""
        hashes = self._tenant_keys.get(tenant_id, [])
        return [
            {
                "key_id": self._keys[h].key_id,
                "name": self._keys[h].name,
                "tier": self._keys[h].tier,
                "is_test": self._keys[h].is_test,
                "is_active": self._keys[h].is_active,
                "created_at": self._keys[h].created_at,
                "last_used_at": self._keys[h].last_used_at,
            }
            for h in hashes if h in self._keys
        ]

    def update_tier(self, tenant_id: str, new_tier: str):
        """Update tier for all active keys of a tenant."""
        for h in self._tenant_keys.get(tenant_id, []):
            if h in self._keys and self._keys[h].is_active:
                self._keys[h].tier = new_tier
        logger.info(f"ApiKeyManager: updated tenant '{tenant_id}' to tier '{new_tier}'")

    @staticmethod
    def _hash(raw_key: str) -> str:
        return hashlib.sha256(raw_key.encode()).hexdigest()


# ─── Singleton ──────────────────────────────────────────────

_manager: ApiKeyManager | None = None


def get_api_key_manager() -> ApiKeyManager:
    global _manager
    if _manager is None:
        _manager = ApiKeyManager()
    return _manager
