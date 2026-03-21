"""Tenant management routes — admin-only (requires master API key)."""

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.dependencies import DbSession, MasterKeyAuth
from app.schemas.tenant import TenantCreate, TenantCreated, TenantResponse, TenantUsage, ApiKeyRotated
from app.services.tenant_service import TenantService

router = APIRouter(prefix="/tenants", tags=["tenants"])


@router.post("", response_model=TenantCreated, status_code=status.HTTP_201_CREATED)
async def create_tenant(body: TenantCreate, db: DbSession, _: MasterKeyAuth):
    """
    Register a new consuming product.

    Returns BOTH API keys — production (sk_live_*) and sandbox (sk_test_*).
    These are shown only once — save them.
    """
    existing = await TenantService.get_by_slug(db, body.slug)
    if existing:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=f"Tenant '{body.slug}' already exists")

    tenant, live_key, test_key = await TenantService.create(
        db,
        name=body.name,
        slug=body.slug,
        contact_email=body.contact_email,
        description=body.description,
        config=body.config or {},
        rate_limits=body.rate_limits,
    )

    return TenantCreated(
        id=tenant.id,
        name=tenant.name,
        slug=tenant.slug,
        is_active=tenant.is_active,
        contact_email=tenant.contact_email,
        description=tenant.description,
        config=tenant.config,
        live_api_key=live_key,
        test_api_key=test_key,
        live_key_prefix=tenant.live_key_prefix,
        test_key_prefix=tenant.test_key_prefix,
        created_at=tenant.created_at.isoformat(),
    )


@router.get("/{tenant_id}", response_model=TenantResponse)
async def get_tenant(tenant_id: int, db: DbSession, _: MasterKeyAuth):
    """Get tenant info by ID. Shows key prefixes but NOT raw keys."""
    tenant = await TenantService.get_by_id(db, tenant_id)
    if not tenant:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Tenant not found")

    return TenantResponse(
        id=tenant.id,
        name=tenant.name,
        slug=tenant.slug,
        is_active=tenant.is_active,
        contact_email=tenant.contact_email,
        description=tenant.description,
        config=tenant.config,
        live_key_prefix=tenant.live_key_prefix,
        test_key_prefix=tenant.test_key_prefix,
        created_at=tenant.created_at.isoformat(),
    )


@router.post("/{tenant_id}/api-keys/rotate", response_model=ApiKeyRotated)
async def rotate_api_key(
    tenant_id: int,
    db: DbSession,
    _: MasterKeyAuth,
    environment: str = Query("live", regex="^(live|test)$", description="Which key to rotate: 'live' or 'test'"),
):
    """
    Rotate a tenant's API key. Old key is immediately invalidated.

    Use `?environment=live` (default) or `?environment=test` to specify which key.
    """
    tenant = await TenantService.get_by_id(db, tenant_id)
    if not tenant:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Tenant not found")

    if environment == "test":
        raw_key = await TenantService.rotate_test_key(db, tenant)
        prefix = tenant.test_key_prefix
    else:
        raw_key = await TenantService.rotate_live_key(db, tenant)
        prefix = tenant.live_key_prefix

    return ApiKeyRotated(environment=environment, api_key=raw_key, api_key_prefix=prefix)
