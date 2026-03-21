"""
Tenant Service — API key generation, validation, and tenant CRUD.

Follows Software Factory pattern: standardized, repeatable tenant provisioning.
"""

import secrets

from passlib.hash import bcrypt
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
        """Hash an API key for storage. Uses bcrypt."""
        return bcrypt.hash(raw_key)

    @staticmethod
    def verify_api_key(raw_key: str, hashed: str) -> bool:
        """Verify a raw API key against its hash."""
        return bcrypt.verify(raw_key, hashed)

    @staticmethod
    def key_prefix(raw_key: str) -> str:
        """Extract a displayable prefix from an API key (e.g., 'sk_live_abc...')."""
        parts = raw_key.split("_", 2)
        if len(parts) >= 3:
            return f"{parts[0]}_{parts[1]}_{parts[2][:8]}..."
        return raw_key[:16] + "..."

    # ─── CRUD ───────────────────────────────────────────────

    @classmethod
    async def create(cls, db: AsyncSession, *, name: str, slug: str, **kwargs) -> tuple[Tenant, str]:
        """
        Create a new tenant and generate its API key.

        Returns (tenant, raw_api_key). The raw key is only available at creation time.
        """
        raw_key = cls.generate_api_key()
        tenant = Tenant(
            name=name,
            slug=slug,
            api_key_hash=cls.hash_api_key(raw_key),
            api_key_prefix=cls.key_prefix(raw_key),
            **kwargs,
        )
        db.add(tenant)
        await db.flush()
        return tenant, raw_key

    @classmethod
    async def resolve_by_api_key(cls, db: AsyncSession, raw_key: str) -> Tenant | None:
        """
        Look up a tenant by API key.

        Since API keys are hashed, we can't do a direct DB lookup.
        We use the prefix to narrow candidates, then verify.
        """
        # Extract prefix for narrowing (e.g., "sk_live")
        prefix_type = "_".join(raw_key.split("_")[:2])

        result = await db.execute(
            select(Tenant).where(
                Tenant.api_key_prefix.startswith(prefix_type),
                Tenant.is_active.is_(True),
            )
        )
        candidates = result.scalars().all()

        for tenant in candidates:
            if cls.verify_api_key(raw_key, tenant.api_key_hash):
                return tenant

        return None

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
    async def rotate_api_key(cls, db: AsyncSession, tenant: Tenant) -> str:
        """Generate a new API key for a tenant, invalidating the old one."""
        raw_key = cls.generate_api_key()
        tenant.api_key_hash = cls.hash_api_key(raw_key)
        tenant.api_key_prefix = cls.key_prefix(raw_key)
        await db.flush()
        return raw_key
