"""
Billing API — Cost transparency for tenants.

Optimization-AI: Endpoints to retrieve token usage summaries and forecasts.
"""

from typing import Annotated
from fastapi import APIRouter, Depends
from app.dependencies import DbSession, get_current_tenant
from app.models.tenant import Tenant
from app.services.intelligence.guardian import get_guardian
from app.services.intelligence.guardian import get_guardian

router = APIRouter(prefix="/billing", tags=["billing"])


@router.get("/summary")
async def get_cost_summary(
    tenant: Annotated[Tenant, Depends(get_current_tenant)],
    db: DbSession,
):
    """Get the current month's usage summary and cost forecast."""
    summary = await TokenForecaster.get_tenant_summary(db, tenant.id)
    return summary
