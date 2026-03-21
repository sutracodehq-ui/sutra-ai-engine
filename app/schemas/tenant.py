"""Pydantic schemas for tenant management."""

from pydantic import BaseModel, Field


class TenantCreate(BaseModel):
    """POST /v1/tenants request body."""

    name: str = Field(..., min_length=1, max_length=255)
    slug: str = Field(..., min_length=1, max_length=100, pattern=r"^[a-z0-9-]+$")
    contact_email: str | None = None
    description: str | None = None
    config: dict | None = Field(default_factory=dict)
    rate_limits: dict | None = None


class TenantResponse(BaseModel):
    """Tenant info response."""

    id: int
    name: str
    slug: str
    is_active: bool
    contact_email: str | None = None
    description: str | None = None
    config: dict | None = None
    created_at: str


class TenantCreated(TenantResponse):
    """Response after creating a tenant — includes the raw API key (shown only once)."""

    api_key: str = Field(..., description="The raw API key — save it, it cannot be retrieved again")


class TenantUsage(BaseModel):
    """GET /v1/tenants/{id}/usage response."""

    tenant_id: int
    period: str
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    breakdown: list[dict] = []


class ApiKeyRotated(BaseModel):
    """Response after rotating an API key."""

    api_key: str
    api_key_prefix: str
