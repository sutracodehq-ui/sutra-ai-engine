"""
Billing API Routes — API key management, usage stats, and tier info.

POST /v1/billing/keys          — Generate new API key
GET  /v1/billing/keys          — List tenant's API keys
POST /v1/billing/keys/revoke   — Revoke an API key
POST /v1/billing/keys/rotate   — Rotate an API key
GET  /v1/billing/usage         — Get usage stats
GET  /v1/billing/usage/today   — Get today's usage
GET  /v1/billing/tiers         — Get all tier info (pricing page)
POST /v1/billing/execute       — Metered agent execution
"""

import logging
from typing import Optional

from fastapi import APIRouter, Header, Query
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1/billing", tags=["billing"])


# ─── Request/Response Models ────────────────────────────────

class GenerateKeyRequest(BaseModel):
    tenant_id: str
    tier: str = Field(default="free", description="free, starter, pro, enterprise")
    name: str = ""
    is_test: bool = False


class RevokeKeyRequest(BaseModel):
    key_id: str = Field(..., description="Key ID prefix (e.g., sk_live_AbCdEf12)")


class RotateKeyRequest(BaseModel):
    tenant_id: str
    old_key_id: str


class ExecuteRequest(BaseModel):
    agent_id: str = Field(..., description="Agent identifier (e.g., copywriter, seo)")
    prompt: str = Field(..., description="Input prompt for the agent")
    context: dict | None = None


# ─── API Key Management ─────────────────────────────────────

@router.post("/keys")
async def generate_key(request: GenerateKeyRequest):
    """Generate a new API key for a tenant."""
    from app.services.billing.api_keys import get_api_key_manager
    manager = get_api_key_manager()

    raw_key = manager.generate(
        tenant_id=request.tenant_id,
        tier=request.tier,
        name=request.name,
        is_test=request.is_test,
    )

    return {
        "api_key": raw_key,
        "message": "Store this key securely. It won't be shown again.",
        "tier": request.tier,
        "is_test": request.is_test,
    }


@router.get("/keys")
async def list_keys(tenant_id: str = Query(...)):
    """List all API keys for a tenant (safe info only)."""
    from app.services.billing.api_keys import get_api_key_manager
    manager = get_api_key_manager()
    return {"keys": manager.list_keys(tenant_id)}


@router.post("/keys/revoke")
async def revoke_key(request: RevokeKeyRequest):
    """Revoke an API key."""
    from app.services.billing.api_keys import get_api_key_manager
    manager = get_api_key_manager()
    success = manager.revoke(request.key_id)
    return {"revoked": success}


@router.post("/keys/rotate")
async def rotate_key(request: RotateKeyRequest):
    """Rotate an API key (generate new, revoke old)."""
    from app.services.billing.api_keys import get_api_key_manager
    manager = get_api_key_manager()
    new_key = manager.rotate(request.tenant_id, request.old_key_id)
    if not new_key:
        return {"error": "Key not found"}
    return {
        "new_api_key": new_key,
        "message": "Old key revoked. Store this new key securely.",
    }


# ─── Usage & Analytics ──────────────────────────────────────

@router.get("/usage")
async def get_usage(
    tenant_id: str = Query(...),
    days: int = Query(default=30, le=90),
):
    """Get usage summary for billing dashboard."""
    from app.services.billing.usage_tracker import get_usage_tracker
    tracker = get_usage_tracker()
    return tracker.get_usage_summary(tenant_id, days)


@router.get("/usage/today")
async def get_today_usage(tenant_id: str = Query(...)):
    """Get today's usage (for real-time dashboard)."""
    from app.services.billing.usage_tracker import get_usage_tracker
    tracker = get_usage_tracker()
    return tracker.get_daily_usage(tenant_id)


# ─── Tier Info ──────────────────────────────────────────────

@router.get("/tiers")
async def get_tiers():
    """Get all tier info for pricing page."""
    from app.services.billing.rate_limiter import get_rate_limiter
    limiter = get_rate_limiter()
    return {"tiers": limiter.all_tiers()}


# ─── Metered Execution ──────────────────────────────────────

@router.post("/execute")
async def metered_execute(
    request: ExecuteRequest,
    x_api_key: str = Header(..., alias="X-API-Key"),
):
    """
    Execute an agent with billing.
    Requires X-API-Key header.
    
    This is the primary monetized endpoint.
    """
    from app.services.billing.gateway import get_billing_gateway
    gateway = get_billing_gateway()

    result = await gateway.execute(
        api_key=x_api_key,
        agent_id=request.agent_id,
        prompt=request.prompt,
        context=request.context,
    )

    return result
