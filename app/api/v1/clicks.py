"""Click Shield API endpoints."""

import uuid
from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_db, get_redis, get_current_tenant
from app.schemas.click_shield import ClickTrackRequest, ClickTrackResponse
from app.services.intelligence.click_scorer import ClickScorerService
from app.models.click_log import ClickLog

router = APIRouter(prefix="/clicks", tags=["click-shield"])

@router.post("/track", response_model=ClickTrackResponse)
async def track_click(
    payload: ClickTrackRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
    redis = Depends(get_redis),
    tenant = Depends(get_current_tenant)
):
    """
    Track and score a click event for fraud.
    
    This endpoint is designed to be called by the Click Shield JS Pixel 
    or directly from the main marketing tool's backend.
    """
    # 1. Initialize Scorer
    scorer = ClickScorerService(redis)
    
    # Use IP from request if not provided
    if not payload.ip:
        payload.ip = request.client.host
    
    # 2. Score the click
    score, action, reasons = await scorer.score(payload)
    
    # 3. Log to Database for audit/ML
    click_id = str(uuid.uuid4())
    log_entry = ClickLog(
        tenant_id=tenant.id,
        ad_id=payload.ad_id,
        ip_address=payload.ip,
        user_agent=payload.client_data.ua,
        referrer=payload.referrer,
        fraud_score=score,
        is_blocked=(action == "BLOCK"),
        action=action,
        reasons=[r.dict() for r in reasons],
        client_data=payload.client_data.dict()
    )
    
    db.add(log_entry)
    await db.commit()
    
    return ClickTrackResponse(
        click_id=click_id,
        fraud_score=score,
        is_blocked=(action == "BLOCK"),
        action=action,
        reasons=reasons,
        request_id=str(uuid.uuid4())
    )

@router.get("/report", response_model=dict)
async def get_click_report(
    db: AsyncSession = Depends(get_db),
    tenant = Depends(get_current_tenant)
):
    """
    Generate a fraud analysis report for the current tenant.
    """
    from sqlalchemy import select, func
    from app.models.click_log import ClickLog
    
    # Simple aggregation for now
    stmt = (
        select(
            func.count(ClickLog.id).label("total"),
            func.sum(case((ClickLog.is_blocked == True, 1), else_=0)).label("blocked"),
            func.avg(ClickLog.fraud_score).label("avg_score")
        )
        .where(ClickLog.tenant_id == tenant.id)
    )
    # Wait, I need to import case from sqlalchemy
    from sqlalchemy import case
    
    result = await db.execute(stmt)
    row = result.fetchone()
    
    total = row.total or 0
    blocked = row.blocked or 0
    
    return {
        "total_clicks": total,
        "blocked_clicks": blocked,
        "fraud_percentage": round((blocked / total * 100), 2) if total > 0 else 0,
        "status": "active"
    }
from app.models.click_feedback import ClickFeedback
from pydantic import BaseModel

class FeedbackRequest(BaseModel):
    is_fraud: bool
    reason: str | None = None
    comment: str | None = None

@router.post("/{click_id}/feedback", response_model=dict)
async def submit_click_feedback(
    click_id: int,
    payload: FeedbackRequest,
    db: AsyncSession = Depends(get_db),
    tenant = Depends(get_current_tenant)
):
    """
    Submit human feedback for a click event to improve the self-learning model.
    """
    # 1. Verify click belongs to tenant
    stmt = select(ClickLog).where(ClickLog.id == click_id, ClickLog.tenant_id == tenant.id)
    result = await db.execute(stmt)
    click = result.scalar_one_or_none()
    
    if not click:
        raise HTTPException(status_code=404, detail="Click event not found")
        
    # 2. Create feedback entry
    feedback = ClickFeedback(
        click_id=click_id,
        is_fraud=payload.is_fraud,
        reason=payload.reason,
        comment=payload.comment
    )
    
    db.add(feedback)
    await db.commit()
    
    return {"status": "success", "message": "Feedback recorded. The AI model will be updated in the next learning cycle."}
