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
    live_key_prefix: str = Field(..., description="Production key prefix (e.g., sk_live_abc12345...)")
    test_key_prefix: str = Field(..., description="Sandbox key prefix (e.g., sk_test_def67890...)")
    created_at: str


class TenantCreated(BaseModel):
    """Response after creating a tenant — includes BOTH raw API keys (shown only once)."""

    id: int
    name: str
    slug: str
    is_active: bool
    contact_email: str | None = None
    description: str | None = None
    config: dict | None = None
    live_api_key: str = Field(..., description="Production API key (sk_live_*) — save it, cannot be retrieved again")
    test_api_key: str = Field(..., description="Sandbox API key (sk_test_*) — save it, cannot be retrieved again")
    live_key_prefix: str
    test_key_prefix: str
    created_at: str


class TenantUsage(BaseModel):
    """GET /v1/tenants/{id}/usage response."""

    tenant_id: int
    period: str
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    breakdown: list[dict] = []


class ApiKeyRotated(BaseModel):
    """Response after rotating an API key."""

    environment: str = Field(..., description="'live' or 'test'")
    api_key: str = Field(..., description="New raw API key — save it, cannot be retrieved again")
    api_key_prefix: str
