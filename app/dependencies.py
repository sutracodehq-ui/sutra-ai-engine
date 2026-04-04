"""
FastAPI dependency injection — DB sessions, auth, Redis.

All request-scoped dependencies are defined here as FastAPI `Depends()` callables.
"""

from typing import Annotated, AsyncGenerator

from fastapi import Depends, HTTPException, status
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


# ─── Security Scheme (Swagger "Authorize" button) ──────────────

from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

_bearer_scheme = HTTPBearer(
    scheme_name="Bearer Auth",
    description="Enter your API key (sk_live_*, sk_test_*) or Master key (sk_master_*).",
    auto_error=True,
)


# ─── Tenant Auth (API Key) ─────────────────────────────────────

async def get_current_tenant(
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(_bearer_scheme)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """
    Extract and validate API key from Authorization header.
    Returns the resolved Tenant model with `_api_environment` set to 'live' or 'test'.

    Auth flow:
    1. Try resolving as SSO JWT (for Sutra-Identity integration)
    2. O(1) indexed hash lookup on api_keys table (checks expiry + active status)
    """
    from app.services.tenant_service import TenantService

    api_key_or_token = credentials.credentials
    if not api_key_or_token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token is required")

    # 1. Try resolving as SSO JWT
    if "." in api_key_or_token and len(api_key_or_token) > 50:
        from app.lib.auth.jwt_validator import JwtValidator
        validator = JwtValidator()
        payload = validator.decode_token(api_key_or_token)
        
        if payload and "org_id" in payload:
            from sqlalchemy import select
            from app.models.tenant import Tenant
            identity_org_id = payload["org_id"]
            
            res = await db.execute(select(Tenant).where(Tenant.identity_org_id == identity_org_id))
            tenant = res.scalar_one_or_none()
            
            if tenant:
                if not tenant.is_active:
                    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Tenant is suspended")
                tenant._api_environment = "live"  # SSO is always live
                return tenant

    # 2. O(1) indexed lookup on api_keys table
    tenant, environment, api_key_record = await TenantService.resolve_by_api_key(db, api_key_or_token)
    if tenant is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key or SSO token")

    if not tenant.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Tenant is suspended")

    tenant._api_environment = environment
    return tenant


async def require_master_key(
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(_bearer_scheme)],
    settings: Annotated[Settings, Depends(get_settings)],
):
    """Validate the master admin API key for tenant management endpoints."""
    api_key = credentials.credentials
    if api_key != settings.master_api_key:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Master API key required")
    return True


# ─── Type Aliases for Clean Route Signatures ──────────────────

DbSession = Annotated[AsyncSession, Depends(get_db)]
CurrentTenant = Annotated["Tenant", Depends(get_current_tenant)]  # noqa: F821
MasterKeyAuth = Annotated[bool, Depends(require_master_key)]

