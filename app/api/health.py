"""Health check endpoints — /health (liveness), /ready (readiness), /health/full, /metrics."""

import asyncio

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_db, get_redis
from app.schemas.common import HealthResponse
from app.services.platform_status import (
    build_full_status,
    build_metrics_snapshot,
    collect_driver_and_circuit_snapshot,
    probe_qdrant,
    probe_migrations,
    probe_redis,
    probe_tenants,
)
from app.services.platform_status import probe_database as probe_db

router = APIRouter(tags=["health"])

_HEALTH_RESPONSES = {
    200: {"description": "OK — see response body."},
    503: {"description": "Degraded — one or more readiness checks failed."},
}


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness probe",
    description=(
        "Kubernetes-style **liveness**: confirms the API process is running. "
        "Does not open database or Redis connections. **No authentication.**"
    ),
    responses={200: {"description": "`{ \"status\": \"ok\", \"service\": \"sutra-ai\" }`"}},
)
async def liveness():
    """Kubernetes liveness probe — process is up (no I/O)."""
    return HealthResponse()


@router.get(
    "/ready",
    summary="Readiness probe",
    description=(
        "Kubernetes-style **readiness**: PostgreSQL (`SELECT 1`), Redis `PING`, "
        "Alembic version row, count of active tenants, Qdrant heartbeat (informational only), "
        "and a per-driver **configured** summary. "
        "**503** if database, Redis, migrations fail, tenants table is missing, or there are zero active tenants. "
        "Qdrant being down does **not** fail readiness. **No authentication.**"
    ),
    responses=_HEALTH_RESPONSES,
)
async def readiness(db: AsyncSession = Depends(get_db), redis=Depends(get_redis)):
    """
    Kubernetes readiness — DB, Redis, migrations, tenants, Qdrant (informational).

    Returns 200 if critical dependencies pass, 503 otherwise.
    Qdrant failure does not fail readiness (optional for RAG-heavy installs).
    """
    checks = {
        "database": "connected",
        "redis": "connected",
        "migrations": "unknown",
        "tenants": 0,
        "qdrant": "unknown",
        "drivers_summary": {},
    }
    is_healthy = True

    db_task = probe_db(db)
    redis_task = probe_redis(redis)
    mig_task = probe_migrations(db)
    ten_task = probe_tenants(db)
    qdrant_task = probe_qdrant()

    db_r, redis_r, mig_r, ten_r, qdrant_r = await asyncio.gather(
        db_task, redis_task, mig_task, ten_task, qdrant_task
    )

    if db_r.get("status") != "ok":
        checks["database"] = "disconnected"
        is_healthy = False
    if redis_r.get("status") != "ok":
        checks["redis"] = "disconnected"
        is_healthy = False

    if mig_r.get("status") == "ok":
        checks["migrations"] = f"current ({mig_r.get('version')})"
    else:
        checks["migrations"] = mig_r.get("error", "unknown")
        is_healthy = False

    if ten_r.get("status") == "ok":
        checks["tenants"] = ten_r.get("active_tenants", 0)
        if checks["tenants"] == 0:
            is_healthy = False
    else:
        checks["tenants"] = ten_r.get("error", "error")
        is_healthy = False

    if qdrant_r.get("status") == "ok":
        checks["qdrant"] = "ok"
    elif qdrant_r.get("status") == "skipped":
        checks["qdrant"] = f"skipped ({qdrant_r.get('reason', '')})"
    else:
        checks["qdrant"] = f"error: {qdrant_r.get('error', qdrant_r)}"

    try:
        checks["drivers_summary"] = {
            k: v.get("configured")
            for k, v in collect_driver_and_circuit_snapshot().get("drivers", {}).items()
            if isinstance(v, dict) and "configured" in v
        }
    except Exception:
        checks["drivers_summary"] = {}

    status_code = 200 if is_healthy else 503
    return {
        "status": "ok" if is_healthy else "degraded",
        **checks,
    } | ({"_http_status": status_code} if not is_healthy else {})


@router.get(
    "/health/full",
    summary="Deep health (all components)",
    description=(
        "Full platform status in one JSON object: **components** (database, redis, migrations, "
        "tenants, Qdrant, Ollama tags vs configured models, each LLM vendor’s model catalog when a key exists), "
        "**system** (CPU count, load average, process RSS), **limits** (admission, VPS profiles, rate limits, timeouts), "
        "**drivers** / **circuit_breaker** / **brain_queue** / **local_stream_admission**, and **api_keys_present** "
        "(booleans only — never secret values). "
        "**503** only when PostgreSQL, Redis, or migrations are unhealthy. **No authentication.**"
    ),
    responses={
        200: {"description": "JSON document; top-level `status` may be `ok`, `degraded`, or `unhealthy`."},
        503: {"description": "Critical failure: database, Redis, or migrations."},
    },
)
async def health_full(db: AsyncSession = Depends(get_db), redis=Depends(get_redis)):
    """
    Deep health: database, Redis, migrations, tenants, every configured LLM endpoint,
    Ollama model tags, Qdrant, CPU/load/process memory, admission limits, circuit breakers.

    503 only when critical path fails (database, redis, or migrations).
    """
    payload = await build_full_status(db, redis)
    critical = (
        payload.get("components", {}).get("database", {}).get("status") == "ok"
        and payload.get("components", {}).get("redis", {}).get("status") == "ok"
        and payload.get("components", {}).get("migrations", {}).get("status") == "ok"
    )
    code = 200 if critical else 503
    if not critical:
        payload["status"] = "unhealthy"
    return JSONResponse(content=payload, status_code=code)


@router.get(
    "/metrics",
    summary="Metrics snapshot (JSON)",
    description=(
        "Same rich snapshot shape as **`GET /health/full`** (see that operation), but always returns **HTTP 200** "
        "with the full JSON body so scrapers and dashboards always receive data. "
        "Use for Grafana/JSON exporters or internal SRE boards. **No authentication.**"
    ),
    responses={
        200: {"description": "JSON metrics document (same schema as `/health/full` when healthy)."},
    },
)
async def metrics_json(db: AsyncSession = Depends(get_db), redis=Depends(get_redis)):
    """
    Full metrics snapshot (JSON) for dashboards or external Prometheus JSON exporters.

    Does not return secret values — only booleans for which API keys are set.
    """
    payload = await build_metrics_snapshot(db, redis)
    return JSONResponse(content=payload, status_code=200)
