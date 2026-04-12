"""
Feedback API Routes — FastAPI endpoints for user feedback.

POST /v1/feedback — Record thumbs up/down on AI response
GET  /v1/feedback/stats — Get per-agent satisfaction metrics
"""

import logging

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession
from app.db.session import get_db

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1/feedback", tags=["feedback"])


class FeedbackRequest(BaseModel):
    """Request body for recording feedback."""
    agent_type: str = Field(..., description="Agent identifier (e.g. 'seo', 'stock_analyzer')")
    prompt: str = Field(..., description="The original user prompt")
    response: str = Field(..., description="The AI response")
    is_positive: bool = Field(..., description="True = 👍, False = 👎")
    quality_score: Optional[float] = Field(None, ge=1, le=5, description="Optional 1-5 star rating")
    feedback_text: Optional[str] = Field(None, description="Optional written feedback")
    user_id: Optional[str] = None
    brand_id: Optional[str] = None
    system_prompt_version: Optional[int] = None


class FeedbackResponse(BaseModel):
    """Response after recording feedback."""
    id: int
    agent_type: str
    is_positive: bool
    message: str


@router.post("", response_model=FeedbackResponse)
async def record_feedback(
    request: FeedbackRequest,
    db: AsyncSession = Depends(get_db),
):
    """Record user feedback on an AI response."""
    from app.services.intelligence.feedback_collector import FeedbackCollector

    collector = FeedbackCollector(db)
    feedback = await collector.record(
        agent_type=request.agent_type,
        prompt=request.prompt,
        response=request.response,
        is_positive=request.is_positive,
        quality_score=request.quality_score,
        feedback_text=request.feedback_text,
        user_id=request.user_id,
        brand_id=request.brand_id,
        system_prompt_version=request.system_prompt_version,
    )

    emoji = "👍" if request.is_positive else "👎"
    return FeedbackResponse(
        id=feedback.id,
        agent_type=feedback.agent_type,
        is_positive=feedback.is_positive,
        message=f"{emoji} Feedback recorded for {request.agent_type}",
    )


@router.get("/stats")
async def get_feedback_stats(
    agent_type: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    """Get feedback statistics, optionally filtered by agent."""
    from app.services.intelligence.feedback_collector import FeedbackCollector

    collector = FeedbackCollector(db)

    if agent_type:
        return await collector.get_agent_stats(agent_type)
    return await collector.get_all_stats()
