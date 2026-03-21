"""Task + Feedback API routes."""

from fastapi import APIRouter, HTTPException, status
from sqlalchemy import select

from app.dependencies import CurrentTenant, DbSession
from app.models.agent_feedback import AgentFeedback
from app.models.ai_task import AiTask
from app.schemas.task import FeedbackRequest, FeedbackResponse, TaskResponse

router = APIRouter(prefix="/tasks", tags=["tasks"])


@router.get("/{task_id}", response_model=TaskResponse)
async def get_task(task_id: int, tenant: CurrentTenant, db: DbSession):
    """Get task status and result."""
    result = await db.execute(
        select(AiTask).where(AiTask.id == task_id, AiTask.tenant_id == tenant.id)
    )
    task = result.scalar_one_or_none()
    if not task:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Task not found")

    return TaskResponse(
        id=task.id,
        tenant_id=task.tenant_id,
        conversation_id=task.conversation_id,
        agent_type=task.agent_type,
        status=task.status,
        prompt=task.prompt,
        result=task.result,
        tokens_used=task.tokens_used,
        driver_used=task.driver_used,
        model_used=task.model_used,
        error=task.error,
        created_at=task.created_at.isoformat(),
        updated_at=task.updated_at.isoformat(),
    )


@router.post("/{task_id}/feedback", response_model=FeedbackResponse)
async def submit_feedback(task_id: int, body: FeedbackRequest, tenant: CurrentTenant, db: DbSession):
    """Submit feedback on a completed task — drives the self-learning loop."""
    result = await db.execute(
        select(AiTask).where(AiTask.id == task_id, AiTask.tenant_id == tenant.id)
    )
    task = result.scalar_one_or_none()
    if not task:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Task not found")
    if task.status != "completed":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Can only provide feedback on completed tasks")

    quality_score = AgentFeedback.score_for_signal(body.signal)

    feedback = AgentFeedback(
        ai_task_id=task.id,
        tenant_id=tenant.id,
        agent_type=task.agent_type,
        signal=body.signal,
        user_edit=body.user_edit,
        quality_score=quality_score,
        comment=body.comment,
    )
    db.add(feedback)
    await db.flush()

    return FeedbackResponse(
        id=feedback.id,
        ai_task_id=feedback.ai_task_id,
        signal=feedback.signal,
        quality_score=feedback.quality_score,
        created_at=feedback.created_at.isoformat(),
    )
