"""Pydantic schemas for task and feedback endpoints."""

from pydantic import BaseModel, Field


class TaskResponse(BaseModel):
    """GET /v1/tasks/{id} response."""

    id: int
    tenant_id: int
    conversation_id: int | None = None
    agent_type: str
    status: str
    prompt: str
    result: dict | None = None
    tokens_used: int = 0
    driver_used: str | None = None
    model_used: str | None = None
    error: str | None = None
    created_at: str
    updated_at: str


class FeedbackRequest(BaseModel):
    """POST /v1/tasks/{id}/feedback request body."""

    signal: str = Field(..., pattern=r"^(accepted|edited|rejected|regenerated)$")
    user_edit: dict | None = Field(None, description="What the user changed: {field: {original, edited}}")
    comment: str | None = None


class FeedbackResponse(BaseModel):
    """Feedback submission response."""

    id: int
    ai_task_id: int
    signal: str
    quality_score: float
    created_at: str
