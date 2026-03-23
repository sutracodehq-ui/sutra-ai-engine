"""Pydantic schemas for chat endpoints."""

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """POST /v1/chat request body."""

    agent_type: str = Field(..., description="Agent to use: copywriter, seo, social_media, email_campaign, etc.")
    prompt: str = Field(..., min_length=1, max_length=10000, description="The prompt to send to the agent")
    voice_profile_id: int | None = Field(None, description="Optional voice profile for tone/style")
    options: dict | None = Field(default_factory=dict, description="Driver-level options (temperature, max_tokens, etc.)")
    external_user_id: str | None = Field(None, description="Your app's user ID for tracking")
    metadata: dict | None = Field(default_factory=dict, description="Arbitrary metadata to attach to the task")


class ChatResponse(BaseModel):
    """Chat completion response."""

    task_id: int
    status: str
    agent_type: str
    result: dict | None = None
    suggestions: list[str] | None = Field(default_factory=list, description="Proactive follow-up suggestions or next steps")
    tokens_used: int = 0
    driver_used: str | None = None
    model_used: str | None = None


class ConversationCreate(BaseModel):
    """POST /v1/conversations request body."""

    agent_type: str
    external_user_id: str | None = None
    metadata: dict | None = Field(default_factory=dict)


class ConversationMessageRequest(BaseModel):
    """POST /v1/conversations/{id}/messages request body."""

    prompt: str = Field(..., min_length=1, max_length=10000)
    voice_profile_id: int | None = None
    options: dict | None = Field(default_factory=dict)


class ConversationResponse(BaseModel):
    """GET /v1/conversations/{id} response."""

    id: int
    agent_type: str
    external_user_id: str | None = None
    metadata: dict | None = None
    messages: list[ChatResponse] = []
    created_at: str
