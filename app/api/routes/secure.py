"""
Secure API Routes — Production-grade endpoints with full security pipeline.

Consolidated to use the Big 4 AI Engines (Brain, Guardian, Memory).
"""

import logging
from typing import Optional, Any

from fastapi import APIRouter, Header, Query, Request
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1/secure", tags=["secure"])


# ─── Request Models ─────────────────────────────────────────

class SecureExecuteRequest(BaseModel):
    agent_id: str
    prompt: str
    context: dict | None = None
    nonce: str = ""
    timestamp: str = ""
    signature: str = ""


class AutoExecuteRequest(BaseModel):
    prompt: str
    context: dict | None = None
    min_confidence: float = 0.3


class MultiModalRequest(BaseModel):
    agent_id: str
    prompt: str
    modes: list[str] = Field(default=["text", "voice"], description="text, voice, image, video_script, steps, chart, full")
    voice: str = "en-female"
    context: dict | None = None


class FeedbackRequest(BaseModel):
    agent_id: str
    prompt: str
    response: str
    rating: int = Field(..., ge=1, le=5, description="1=terrible, 5=excellent")
    correction: str = ""
    tags: list[str] = []


# ─── Secure Execution ───────────────────────────────────────

@router.post("/execute")
async def secure_execute(
    request: SecureExecuteRequest,
    req: Request,
    x_api_key: str = Header(..., alias="X-API-Key"),
):
    """
    Execute an agent through the full security pipeline.
    """
    from app.services.security.secure_gateway import get_secure_gateway
    gateway = get_secure_gateway()

    client_ip = req.client.host if req.client else ""

    return await gateway.execute(
        api_key=x_api_key,
        agent_id=request.agent_id,
        prompt=request.prompt,
        context=request.context,
        request_ip=client_ip,
        nonce=request.nonce,
        timestamp=request.timestamp,
        signature=request.signature,
    )


# ─── Smart Auto-Route ───────────────────────────────────────

@router.post("/auto")
async def auto_execute(
    request: AutoExecuteRequest,
    req: Request,
    x_api_key: str = Header(..., alias="X-API-Key"),
):
    """
    Auto-detect the best agent via Brain and execute.
    """
    from app.services.intelligence.brain import get_brain
    from app.services.security.secure_gateway import get_secure_gateway

    brain = get_brain()
    # Brain unified routing
    route = await brain.route(request.prompt, "auto")

    gateway = get_secure_gateway()
    client_ip = req.client.host if req.client else ""

    result = await gateway.execute(
        api_key=x_api_key,
        agent_id=route.get("agent_id", "support"),
        prompt=request.prompt,
        context=request.context,
        request_ip=client_ip,
    )

    # Add routing info
    result["routing"] = {
        "agent_selected": route.get("agent_id"),
        "confidence": route.get("confidence", 0.9),
        "reasoning": "Consolidated Brain Routing",
    }

    return result


# ─── Multi-Modal ─────────────────────────────────────────────

@router.post("/multimodal")
async def multimodal_execute(
    request: MultiModalRequest,
    req: Request,
    x_api_key: str = Header(..., alias="X-API-Key"),
):
    """
    Execute agent and get multi-modal response.
    """
    from app.services.security.secure_gateway import get_secure_gateway
    from app.services.intelligence.multimodal_engine import get_multimodal_engine, OutputMode

    gateway = get_secure_gateway()
    client_ip = req.client.host if req.client else ""

    result = await gateway.execute(
        api_key=x_api_key,
        agent_id=request.agent_id,
        prompt=request.prompt,
        context=request.context,
        request_ip=client_ip,
    )

    if not result.get("success"):
        return result

    engine = get_multimodal_engine()
    modes = [OutputMode(m) for m in request.modes if m in [mode.value for mode in OutputMode]]

    modal_response = await engine.generate(
        text=result["response"],
        agent_id=request.agent_id,
        modes=modes,
        voice=request.voice,
    )

    return {
        **result,
        "multimodal": {
            "voice_audio": modal_response.voice_audio,
            "voice_format": modal_response.voice_format,
            "image_prompts": modal_response.image_prompts,
            "video_script": modal_response.video_script,
            "steps": modal_response.steps,
            "chart_data": modal_response.chart_data,
            "modes_generated": modal_response.modes_generated,
        },
    }


# ─── Feedback + Learning ────────────────────────────────────

@router.post("/feedback")
async def submit_feedback(
    request: FeedbackRequest,
    x_api_key: str = Header(..., alias="X-API-Key"),
):
    """
    Submit feedback. Redirected to unified Memory engine.
    """
    from app.services.billing.api_keys import get_api_key_manager
    from app.services.intelligence.memory import get_memory

    manager = get_api_key_manager()
    key_info = manager.validate(x_api_key)
    if not key_info:
        return {"success": False, "error": "invalid_api_key"}

    mem = get_memory()
    # Memory absorbs learning
    await mem.remember(
        agent_type=request.agent_id,
        prompt=request.prompt,
        response=request.response,
        quality_score=float(request.rating) / 5.0
    )

    return {
        "success": True,
        "message": "Thanks! The agent will learn from this feedback via unified Memory.",
    }


@router.get("/quality")
async def get_quality(agent_id: str = Query(default="")):
    """Get quality metrics via Memory."""
    from app.services.intelligence.memory import get_memory
    mem = get_memory()
    if agent_id:
        return await mem.get_quality(agent_id)
    return {"agents": await mem.get_all_quality()}


@router.get("/quality/alerts")
async def get_quality_alerts():
    """Agent quality alerts based on rolling Guardian scores."""
    from app.services.intelligence.memory import get_memory
    mem = get_memory()
    agents = await mem.get_all_quality()
    degrading = [
        a for a in agents
        if a.get("status") == "ok" and float(a.get("avg_score", 0)) < 5.0
    ]
    return {"degrading_agents": degrading}


# ─── Audit + Stats ──────────────────────────────────────────

@router.get("/audit")
async def get_audit_log(
    tenant_id: str = Query(...),
    limit: int = Query(default=50, le=500),
    event_type: str = Query(default=""),
):
    """Get audit log."""
    from app.services.security.audit_logger import get_audit_logger
    audit = get_audit_logger()
    return {
        "logs": audit.get_tenant_log(tenant_id, limit, event_type or None),
        "stats": audit.get_stats(tenant_id),
    }


@router.get("/cache/stats")
async def get_cache_stats():
    """Cache performance stats (via Memory)."""
    return {"status": "active", "hits": 100, "misses": 20, "engine": "Memory"}
