"""Tenant management routes — admin-only (requires master API key)."""

from fastapi import APIRouter, Depends, HTTPException, status

from app.dependencies import DbSession, MasterKeyAuth
from app.schemas.tenant import TenantCreate, TenantCreated, TenantResponse, TenantUsage, ApiKeyRotated
from app.services.tenant_service import TenantService

router = APIRouter(prefix="/tenants", tags=["tenants"])


@router.post("", response_model=TenantCreated, status_code=status.HTTP_201_CREATED)
async def create_tenant(body: TenantCreate, db: DbSession, _: MasterKeyAuth):
    """Register a new consuming product. Returns API key (shown only once)."""
    existing = await TenantService.get_by_slug(db, body.slug)
    if existing:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=f"Tenant '{body.slug}' already exists")

    tenant, raw_key = await TenantService.create(
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
        api_key=raw_key,
        created_at=tenant.created_at.isoformat(),
    )


@router.get("/{tenant_id}", response_model=TenantResponse)
async def get_tenant(tenant_id: int, db: DbSession, _: MasterKeyAuth):
    """Get tenant info by ID."""
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
        created_at=tenant.created_at.isoformat(),
    )


@router.post("/{tenant_id}/api-keys", response_model=ApiKeyRotated)
async def rotate_api_key(tenant_id: int, db: DbSession, _: MasterKeyAuth):
    """Rotate a tenant's API key. Old key is immediately invalidated."""
    tenant = await TenantService.get_by_id(db, tenant_id)
    if not tenant:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Tenant not found")

    raw_key = await TenantService.rotate_api_key(db, tenant)
    return ApiKeyRotated(api_key=raw_key, api_key_prefix=tenant.api_key_prefix)
