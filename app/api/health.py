"""Health check endpoints — /health (liveness), /ready (readiness)."""

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_db, get_redis
from app.schemas.common import HealthResponse, ReadyResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def liveness():
    """Kubernetes liveness probe — always returns 200."""
    return HealthResponse()


@router.get("/ready", response_model=ReadyResponse)
async def readiness(db: AsyncSession = Depends(get_db), redis=Depends(get_redis)):
    """Kubernetes readiness probe — checks DB + Redis connectivity."""
    db_status = "connected"
    redis_status = "connected"
    chromadb_status = "unknown"

    try:
        await db.execute(text("SELECT 1"))
    except Exception:
        db_status = "disconnected"

    try:
        await redis.ping()
    except Exception:
        redis_status = "disconnected"

    status = "ok" if db_status == "connected" and redis_status == "connected" else "degraded"

    return ReadyResponse(status=status, database=db_status, redis=redis_status, chromadb=chromadb_status)
