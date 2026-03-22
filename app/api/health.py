"""Health check endpoints — /health (liveness), /ready (readiness)."""

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_db, get_redis
from app.schemas.common import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def liveness():
    """Kubernetes liveness probe — always returns 200."""
    return HealthResponse()


@router.get("/ready")
async def readiness(db: AsyncSession = Depends(get_db), redis=Depends(get_redis)):
    """
    Kubernetes readiness probe — checks DB, Redis, migrations, and tenants.

    Returns 200 if all checks pass, 503 if any critical check fails.
    """
    checks = {
        "database": "connected",
        "redis": "connected",
        "migrations": "unknown",
        "tenants": 0,
    }
    is_healthy = True

    # ─── Database ────────────────────────────────────────
    try:
        await db.execute(text("SELECT 1"))
    except Exception:
        checks["database"] = "disconnected"
        is_healthy = False

    # ─── Redis ───────────────────────────────────────────
    try:
        await redis.ping()
    except Exception:
        checks["redis"] = "disconnected"
        is_healthy = False

    # ─── Migrations ──────────────────────────────────────
    try:
        result = await db.execute(
            text("SELECT version_num FROM alembic_version LIMIT 1")
        )
        row = result.scalar_one_or_none()
        if row:
            checks["migrations"] = f"current ({row})"
        else:
            checks["migrations"] = "no version found"
            is_healthy = False
    except Exception:
        checks["migrations"] = "alembic_version table missing — run migrations"
        is_healthy = False

    # ─── Tenants ─────────────────────────────────────────
    try:
        result = await db.execute(
            text("SELECT COUNT(*) FROM tenants WHERE is_active = true")
        )
        count = result.scalar_one()
        checks["tenants"] = count
        if count == 0:
            is_healthy = False
    except Exception:
        checks["tenants"] = "table missing"
        is_healthy = False

    status_code = 200 if is_healthy else 503
    return {
        "status": "ok" if is_healthy else "degraded",
        **checks,
    } | ({"_http_status": status_code} if not is_healthy else {})

