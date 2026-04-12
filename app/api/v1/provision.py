"""
Provisioning API — Atomic cross-service setup.

Identity-AI: Secure endpoint for Sutra-Identity to provision 
new organizations in the AI Engine.
"""

from typing import Annotated
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.dependencies import DbSession, MasterKeyAuth
from app.services.tenant_service import TenantService
from app.services.voice_profile_service import VoiceProfileService

router = APIRouter(prefix="/provision", tags=["provisioning"])


class OrgProvisionRequest(BaseModel):
    """Request from Sutra-Identity to onboard a new org."""
    identity_org_id: str = Field(..., description="The UUID from Sutra-Identity")
    name: str = Field(..., description="Organization name")
    slug: str = Field(..., description="Organization slug")
    contact_email: str | None = None


@router.post("/org", status_code=status.HTTP_201_CREATED)
async def provision_org(
    body: OrgProvisionRequest,
    db: DbSession,
    _: MasterKeyAuth
):
    """
    Downstream provisioning: identity service calls this to 
    initialize a workspace in the engine.
    """
    # 1. Check for duplicate mapping
    from sqlalchemy import select
    from app.models.tenant import Tenant
    res = await db.execute(select(Tenant).where(Tenant.identity_org_id == body.identity_org_id))
    if res.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="Org already provisioned")

    # 2. Create Tenant
    tenant, live_key, test_key = await TenantService.create(
        db,
        name=body.name,
        slug=body.slug,
        contact_email=body.contact_email,
        identity_org_id=body.identity_org_id
    )

    # 3. Atomic Setup: Default Voice Profile
    # We assume 'General' profile is needed by default
    await VoiceProfileService.create(
        db,
        tenant_id=tenant.id,
        name="Brand Standard",
        slug="brand-standard",
        description="Default AI voice for the organization",
        is_default=True
    )
    
    await db.commit()

    return {
        "status": "success",
        "tenant_id": tenant.id,
        "identity_org_id": tenant.identity_org_id,
        "live_api_key": live_key,
        "test_api_key": test_key
    }
