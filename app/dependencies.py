"""
FastAPI dependency injection — DB sessions, auth, Redis.

All request-scoped dependencies are defined here as FastAPI `Depends()` callables.

Auth flow:
  1. HTTPBearer extracts token from Authorization header
  2. Master key → resolves first tenant with tier=master, scopes=["*"]
  3. SSO JWT → resolves tenant by identity_org_id
  4. API key → O(1) indexed hash lookup on api_keys table
  5. Tier + scopes stored on request.state for middleware enforcement
"""

from typing import Annotated, AsyncGenerator

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
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

_bearer_scheme = HTTPBearer(
    scheme_name="Bearer Auth",
    description="Enter your API key (sk_live_*, sk_test_*) or Master key (sk_master_*).",
    auto_error=True,
)


# ─── Tenant Auth (API Key / Master Key / SSO JWT) ──────────────

async def get_current_tenant(
    request: Request,
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(_bearer_scheme)],
    db: Annotated[AsyncSession, Depends(get_db)],
    settings: Annotated[Settings, Depends(get_settings)],
):
    """
    Unified auth resolver — handles master key, SSO JWT, and tenant API keys.

    Sets request.state._api_tier and ._api_scopes for middleware enforcement.

    Auth priority:
      1. Master key → superadmin (tier=master, scopes=["*"])
      2. SSO JWT → tenant by identity_org_id (tier=standard, scopes=["*"])
      3. API key → O(1) indexed hash lookup (tier + scopes from api_keys table)
    """
    from app.services.tenant_service import TenantService

    token = credentials.credentials
    if not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token is required")

    # ─── 1. Master Key → God Mode ──────────────────────
    if token == settings.master_api_key:
        from sqlalchemy import select
        from app.models.tenant import Tenant

        # Resolve first active tenant for master key to operate on
        result = await db.execute(select(Tenant).where(Tenant.is_active.is_(True)).order_by(Tenant.id).limit(1))
        tenant = result.scalar_one_or_none()

        if not tenant:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="No active tenant found")

        tenant._api_environment = "live"
        tenant._api_tier = "master"
        tenant._api_scopes = ["*"]

        # Set on request.state for middleware
        request.state._api_tier = "master"
        request.state._api_scopes = ["*"]

        return tenant

    # ─── 2. SSO JWT → Tenant by org_id ─────────────────
    if "." in token and len(token) > 50:
        from app.lib.auth.jwt_validator import JwtValidator
        validator = JwtValidator()
        payload = validator.decode_token(token)

        if payload and "org_id" in payload:
            from sqlalchemy import select
            from app.models.tenant import Tenant
            identity_org_id = payload["org_id"]

            res = await db.execute(select(Tenant).where(Tenant.identity_org_id == identity_org_id))
            tenant = res.scalar_one_or_none()

            if tenant:
                if not tenant.is_active:
                    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Tenant is suspended")
                tenant._api_environment = "live"
                tenant._api_tier = "standard"
                tenant._api_scopes = ["*"]

                request.state._api_tier = "standard"
                request.state._api_scopes = ["*"]

                return tenant

    # ─── 3. API Key → O(1) indexed lookup ──────────────
    tenant, environment, api_key_record = await TenantService.resolve_by_api_key(db, token)
    if tenant is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key or SSO token")

    if not tenant.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Tenant is suspended")

    # Extract tier + scopes from the resolved api_key record
    tier = getattr(api_key_record, "tier", "standard")
    scopes = getattr(api_key_record, "scopes", ["*"])

    tenant._api_environment = environment
    tenant._api_tier = tier
    tenant._api_scopes = scopes or ["*"]

    # Set on request.state for scope middleware
    request.state._api_tier = tier
    request.state._api_scopes = scopes or ["*"]

    return tenant


async def require_master_key(
    request: Request,
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(_bearer_scheme)],
    settings: Annotated[Settings, Depends(get_settings)],
):
    """Validate the master admin API key for tenant management endpoints."""
    api_key = credentials.credentials
    if api_key != settings.master_api_key:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Master API key required")

    # Set tier metadata for middleware consistency
    request.state._api_tier = "master"
    request.state._api_scopes = ["*"]

    return True


# ─── Type Aliases for Clean Route Signatures ──────────────────

DbSession = Annotated[AsyncSession, Depends(get_db)]
CurrentTenant = Annotated["Tenant", Depends(get_current_tenant)]  # noqa: F821
MasterKeyAuth = Annotated[bool, Depends(require_master_key)]
