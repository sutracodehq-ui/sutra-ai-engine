"""Pydantic schema for the standardized agent output after filtration."""

from pydantic import BaseModel, Field


class AgentResult(BaseModel):
    """
    Guaranteed output structure from the Response Filtration Engine.

    Every agent response — regardless of LLM quality — is normalized into this shape.
    The frontend can always rely on these fields existing.
    """

    data: dict = Field(default_factory=dict, description="Agent's domain-specific fields (advice, action_items, etc.)")
    suggestions: list[str] = Field(default_factory=list, description="2-3 actionable follow-up suggestions")
    raw: str = Field(default="", description="Original LLM text for debugging/logging")
    parsed: bool = Field(default=True, description="Whether JSON parsing succeeded")
