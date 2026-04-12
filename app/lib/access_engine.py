"""
Access Control Engine — Config-driven, set-theory based permission resolution.

Loads access_control.yaml once at startup and provides O(1) permission checks.

Resolution formula (Set Theory):
    Effective = KeyScopes ∪ TierDefaults
    Access    = Required ⊆ Effective

Software Factory: All rules live in YAML. This engine just evaluates them.
"""

import logging
from functools import lru_cache
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)

# ─── Config Path ────────────────────────────────────────────────

_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "access_control.yaml"


class AccessEngine:
    """
    Config-driven access control engine.

    Singleton — loaded once, cached forever. Hot-reload via `reload()`.
    """

    def __init__(self, config_path: Path | None = None):
        self._path = config_path or _CONFIG_PATH
        self._config: dict = {}
        self._tiers: dict = {}
        self._scopes: dict = {}
        self._route_scopes: dict = {}
        self._valid_scopes: set = set()
        self._load()

    def _load(self) -> None:
        """Load and index the access control config."""
        with open(self._path) as f:
            self._config = yaml.safe_load(f)

        self._tiers = self._config.get("tiers", {})
        self._scopes = self._config.get("scopes", {})
        self._route_scopes = self._config.get("route_scopes", {})

        # Pre-compute valid scope set for O(1) validation
        self._valid_scopes = set(self._scopes.keys())

        logger.info(
            f"AccessEngine: loaded {len(self._tiers)} tiers, "
            f"{len(self._valid_scopes)} scopes, "
            f"{len(self._route_scopes)} route mappings"
        )

    def reload(self) -> None:
        """Hot-reload config without restarting the server."""
        self._load()
        logger.info("AccessEngine: config reloaded")

    # ─── Tier Queries ──────────────────────────────────────────

    def get_tier(self, tier_name: str) -> dict:
        """Get tier config by name. Returns empty dict if not found."""
        return self._tiers.get(tier_name, {})

    def get_tier_scopes(self, tier_name: str) -> list[str]:
        """Get default scopes for a tier."""
        return self.get_tier(tier_name).get("scopes", [])

    def is_tier_protected(self, tier_name: str) -> bool:
        """Check if tier is protected (cannot be created via API)."""
        return self.get_tier(tier_name).get("protected", False)

    def can_access_admin(self, tier_name: str) -> bool:
        """Check if tier has admin access (tenant CRUD, key management)."""
        return self.get_tier(tier_name).get("can_access_admin", False)

    def can_impersonate_tenant(self, tier_name: str) -> bool:
        """Check if tier can use admin key on tenant routes."""
        return self.get_tier(tier_name).get("can_impersonate_tenant", False)

    def get_available_tiers(self, include_protected: bool = False) -> list[str]:
        """List tiers available for key creation."""
        return [
            name for name, cfg in self._tiers.items()
            if include_protected or not cfg.get("protected", False)
        ]

    # ─── Scope Queries ─────────────────────────────────────────

    def get_available_scopes(self) -> dict[str, str]:
        """Get all available scopes with descriptions."""
        return dict(self._scopes)

    def validate_scopes(self, scopes: list[str]) -> list[str]:
        """
        Validate a list of scopes against the canonical list.
        Returns list of invalid scopes (empty = all valid).
        """
        if not scopes:
            return []
        return [s for s in scopes if s not in self._valid_scopes]

    # ─── Route Queries ─────────────────────────────────────────

    def is_public(self, tag: str) -> bool:
        """Check if a route tag is public (no auth required)."""
        route = self._route_scopes.get(tag, {})
        return route.get("public", False)

    def is_route_protected(self, tag: str) -> bool:
        """Check if a route tag is admin-only."""
        route = self._route_scopes.get(tag, {})
        return route.get("protected", False)

    def get_required_scope(self, tag: str, method: str) -> Optional[str]:
        """
        Get the required scope for a route tag + HTTP method.

        GET/HEAD/OPTIONS → "read" scope
        POST/PUT/PATCH/DELETE → "write" scope

        Returns None if no scope mapping exists (allow by default).
        """
        route = self._route_scopes.get(tag, {})

        if route.get("public") or route.get("protected"):
            return None  # Handled by is_public/is_route_protected

        # Map HTTP method to read/write
        if method.upper() in ("GET", "HEAD", "OPTIONS"):
            return route.get("read")
        else:
            return route.get("write")

    # ─── Permission Resolution (Set Theory) ────────────────────

    def resolve(
        self,
        key_scopes: list[str] | None,
        tier: str,
        required_scope: str,
    ) -> bool:
        """
        Core resolution engine — O(1) set intersection check.

        Formula:
            Effective = KeyScopes ∪ TierDefaults
            Access    = Required ⊆ Effective

        Wildcard rules:
            "*" in Effective     → always allowed
            "agents:*" satisfies "agents:read" and "agents:write"
        """
        # Step 1: Compute effective scope set
        tier_scopes = self.get_tier_scopes(tier)
        effective = set(key_scopes or []) | set(tier_scopes)

        # Step 2: Wildcard check (god mode)
        if "*" in effective:
            return True

        # Step 3: Exact match
        if required_scope in effective:
            return True

        # Step 4: Resource wildcard ("agents:*" covers "agents:read")
        resource = required_scope.split(":")[0]
        if f"{resource}:*" in effective:
            return True

        return False

    def check_access(
        self,
        key_scopes: list[str] | None,
        tier: str,
        tag: str,
        method: str,
    ) -> tuple[bool, str]:
        """
        Full access check for a request.

        Returns (allowed: bool, reason: str).
        """
        # Public routes — always allowed
        if self.is_public(tag):
            return True, "public"

        # Protected routes — require admin tier
        if self.is_route_protected(tag):
            if self.can_access_admin(tier):
                return True, "admin_tier"
            return False, f"Route '{tag}' requires admin access"

        # Scoped routes — resolve via set theory
        required = self.get_required_scope(tag, method)
        if required is None:
            # No scope mapping → allow (unmapped routes default to open)
            return True, "no_scope_required"

        if self.resolve(key_scopes, tier, required):
            return True, "scope_granted"

        return False, f"Scope '{required}' required. Your key does not have this permission."


# ─── Singleton ──────────────────────────────────────────────────

_engine: AccessEngine | None = None


def get_access_engine() -> AccessEngine:
    """Get the global AccessEngine instance (loaded once)."""
    global _engine
    if _engine is None:
        _engine = AccessEngine()
    return _engine
