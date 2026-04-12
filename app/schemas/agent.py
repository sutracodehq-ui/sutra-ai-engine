"""Pydantic schemas for agent endpoints."""

from pydantic import BaseModel, Field


class AgentInfo(BaseModel):
    """Agent metadata for GET /v1/agents."""

    identifier: str
    name: str
    domain: str
    description: str
    capabilities: list[str]
    response_schema: list[str]


class AgentRunRequest(BaseModel):
    """POST /v1/agents/{type}/run request body."""

    prompt: str = Field(..., min_length=1, max_length=10000)
    voice_profile_id: int | None = None
    options: dict | None = Field(default_factory=dict)
    external_user_id: str | None = None
    metadata: dict | None = Field(default_factory=dict)


class BatchRunRequest(BaseModel):
    """POST /v1/agents/batch request body."""

    prompt: str = Field(..., min_length=1, max_length=10000)
    agent_types: list[str] = Field(..., min_length=1)
    voice_profile_id: int | None = None
    options: dict | None = Field(default_factory=dict)
    external_user_id: str | None = None
