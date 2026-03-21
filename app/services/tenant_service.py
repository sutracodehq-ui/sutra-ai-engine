"""
Tenant Service — dual API key generation, validation, and tenant CRUD.

Each tenant gets two keys:
  - sk_live_* → production (real LLM calls, billed)
  - sk_test_* → sandbox (mock driver, no billing)
"""

import hashlib
import secrets

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.tenant import Tenant


class TenantService:
    """Factory for tenant lifecycle operations."""

    # ─── API Key Management ─────────────────────────────────

    @staticmethod
    def generate_api_key(prefix: str = "sk_live") -> str:
        """Generate a cryptographically secure API key."""
        random_part = secrets.token_urlsafe(32)
        return f"{prefix}_{random_part}"

    @staticmethod
    def hash_api_key(raw_key: str) -> str:
        """Hash an API key for storage. Uses SHA-256 (keys are already high-entropy)."""
        return hashlib.sha256(raw_key.encode()).hexdigest()

    @staticmethod
    def verify_api_key(raw_key: str, hashed: str) -> bool:
        """Verify a raw API key against its hash."""
        return hashlib.sha256(raw_key.encode()).hexdigest() == hashed

    @staticmethod
    def key_prefix(raw_key: str) -> str:
        """Extract a displayable prefix from an API key (e.g., 'sk_live_abc12345...')."""
        parts = raw_key.split("_", 2)
        if len(parts) >= 3:
            return f"{parts[0]}_{parts[1]}_{parts[2][:8]}..."
        return raw_key[:16] + "..."

    @staticmethod
    def key_environment(raw_key: str) -> str:
        """Detect if a key is live or test from its prefix."""
        if raw_key.startswith("sk_test_"):
            return "test"
        return "live"

    # ─── CRUD ───────────────────────────────────────────────

    @classmethod
    async def create(cls, db: AsyncSession, *, name: str, slug: str, **kwargs) -> tuple[Tenant, str, str]:
        """
        Create a new tenant and generate both API keys.

        Returns (tenant, raw_live_key, raw_test_key).
        Raw keys are only available at creation time — save them.
        """
        raw_live_key = cls.generate_api_key("sk_live")
        raw_test_key = cls.generate_api_key("sk_test")

        tenant = Tenant(
            name=name,
            slug=slug,
            live_key_hash=cls.hash_api_key(raw_live_key),
            live_key_prefix=cls.key_prefix(raw_live_key),
            test_key_hash=cls.hash_api_key(raw_test_key),
            test_key_prefix=cls.key_prefix(raw_test_key),
            **kwargs,
        )
        db.add(tenant)
        await db.flush()
        return tenant, raw_live_key, raw_test_key

    @classmethod
    async def resolve_by_api_key(cls, db: AsyncSession, raw_key: str) -> tuple[Tenant | None, str]:
        """
        Look up a tenant by API key.

        Returns (tenant, environment) where environment is 'live' or 'test'.
        Checks the correct hash column based on the key prefix.
        """
        env = cls.key_environment(raw_key)

        result = await db.execute(
            select(Tenant).where(Tenant.is_active.is_(True))
        )
        candidates = result.scalars().all()

        for tenant in candidates:
            hash_to_check = tenant.test_key_hash if env == "test" else tenant.live_key_hash
            if cls.verify_api_key(raw_key, hash_to_check):
                return tenant, env

        return None, env

    @classmethod
    async def get_by_id(cls, db: AsyncSession, tenant_id: int) -> Tenant | None:
        """Get a tenant by ID."""
        result = await db.execute(select(Tenant).where(Tenant.id == tenant_id))
        return result.scalar_one_or_none()

    @classmethod
    async def get_by_slug(cls, db: AsyncSession, slug: str) -> Tenant | None:
        """Get a tenant by slug."""
        result = await db.execute(select(Tenant).where(Tenant.slug == slug))
        return result.scalar_one_or_none()

    @classmethod
    async def rotate_live_key(cls, db: AsyncSession, tenant: Tenant) -> str:
        """Generate a new production API key, invalidating the old one."""
        raw_key = cls.generate_api_key("sk_live")
        tenant.live_key_hash = cls.hash_api_key(raw_key)
        tenant.live_key_prefix = cls.key_prefix(raw_key)
        await db.flush()
        return raw_key

    @classmethod
    async def rotate_test_key(cls, db: AsyncSession, tenant: Tenant) -> str:
        """Generate a new sandbox API key, invalidating the old one."""
        raw_key = cls.generate_api_key("sk_test")
        tenant.test_key_hash = cls.hash_api_key(raw_key)
        tenant.test_key_prefix = cls.key_prefix(raw_key)
        await db.flush()
        return raw_key
