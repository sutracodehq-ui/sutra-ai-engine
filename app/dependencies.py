"""
FastAPI dependency injection — DB sessions, auth, Redis.

All request-scoped dependencies are defined here as FastAPI `Depends()` callables.
"""

from typing import Annotated, AsyncGenerator

from fastapi import Depends, Header, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import Settings, get_settings

# ─── Database Engine (created once at import) ───────────────────

_engine = None
_session_factory = None


def _get_engine(settings: Settings):
    global _engine, _session_factory
    if _engine is None:
        _engine = create_async_engine(
            settings.database_url,
            echo=settings.debug,
            pool_size=20,
            max_overflow=10,
        )
        _session_factory = async_sessionmaker(_engine, expire_on_commit=False)
    return _engine


async def get_db(settings: Annotated[Settings, Depends(get_settings)]) -> AsyncGenerator[AsyncSession, None]:
    """Yield a scoped async DB session per request."""
    _get_engine(settings)
    async with _session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


# ─── Redis ──────────────────────────────────────────────────────

_redis_pool = None


async def get_redis(settings: Annotated[Settings, Depends(get_settings)]):
    """Get a Redis connection from the pool."""
    global _redis_pool
    if _redis_pool is None:
        import redis.asyncio as aioredis

        _redis_pool = aioredis.from_url(settings.redis_url, decode_responses=True)
    return _redis_pool


# ─── Tenant Auth (API Key) ─────────────────────────────────────

async def get_current_tenant(
    authorization: Annotated[str, Header()],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """
    Extract and validate API key from Authorization header.
    Returns the resolved Tenant model.

    Header format: `Authorization: Bearer sk_live_xxxxxxxx`
    """
    from app.services.tenant_service import TenantService

    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid authorization header format")

    api_key = authorization.removeprefix("Bearer ").strip()
    if not api_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="API key is required")

    tenant = await TenantService.resolve_by_api_key(db, api_key)
    if tenant is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")

    if not tenant.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Tenant is suspended")

    return tenant


async def require_master_key(
    authorization: Annotated[str, Header()],
    settings: Annotated[Settings, Depends(get_settings)],
):
    """Validate the master admin API key for tenant management endpoints."""
    api_key = authorization.removeprefix("Bearer ").strip()
    if api_key != settings.master_api_key:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Master API key required")
    return True


# ─── Type Aliases for Clean Route Signatures ──────────────────

DbSession = Annotated[AsyncSession, Depends(get_db)]
CurrentTenant = Annotated["Tenant", Depends(get_current_tenant)]  # noqa: F821
MasterKeyAuth = Annotated[bool, Depends(require_master_key)]
