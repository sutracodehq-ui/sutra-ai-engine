"""
Secure API Routes — Production-grade endpoints with full security pipeline.

POST /v1/secure/execute         — Secure + metered agent execution
POST /v1/secure/auto            — Smart-routed execution (auto-detect agent)
POST /v1/secure/multimodal      — Multi-modal output (text + voice + image + steps)
POST /v1/secure/feedback        — Submit feedback on agent response
GET  /v1/secure/quality         — Agent quality dashboard
GET  /v1/secure/quality/alerts  — Degrading agents alert
GET  /v1/secure/audit           — Audit log for a tenant
GET  /v1/secure/cache/stats     — Cache performance stats
"""

import logging
from typing import Optional

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
    
    Pipeline: API Key → IP Check → Anti-Replay → HMAC → Injection Guard →
              PII Redact → Rate Limit → Cache → Agent → Audit
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
    Auto-detect the best agent and execute.
    
    Just send a prompt — the smart router picks the right specialist.
    """
    from app.services.optimization.smart_router import get_smart_router
    from app.services.security.secure_gateway import get_secure_gateway

    router_engine = get_smart_router()
    route = router_engine.route_with_fallback(request.prompt, request.min_confidence)

    gateway = get_secure_gateway()
    client_ip = req.client.host if req.client else ""

    result = await gateway.execute(
        api_key=x_api_key,
        agent_id=route.agent_id,
        prompt=request.prompt,
        context=request.context,
        request_ip=client_ip,
    )

    # Add routing info
    result["routing"] = {
        "agent_selected": route.agent_id,
        "confidence": round(route.confidence, 2),
        "reasoning": route.reasoning,
        "alternatives": route.alternatives,
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
    
    Returns text + voice audio + image prompts + video script + steps.
    """
    from app.services.security.secure_gateway import get_secure_gateway
    from app.services.intelligence.multimodal_engine import get_multimodal_engine, OutputMode

    # Execute agent through secure pipeline
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

    # Generate multi-modal outputs
    engine = get_multimodal_engine()
    modes = [OutputMode(m) for m in request.modes if m in OutputMode.__members__.values()]

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
    Submit feedback on an agent response. 
    The agent learns from this for future queries.
    """
    from app.services.billing.api_keys import get_api_key_manager
    from app.services.intelligence.agent_learning import get_agent_learning

    # Validate key
    manager = get_api_key_manager()
    key_info = manager.validate(x_api_key)
    if not key_info:
        return {"success": False, "error": "invalid_api_key"}

    learning = get_agent_learning()
    feedback_id = learning.submit_feedback(
        agent_id=request.agent_id,
        tenant_id=key_info.tenant_id,
        prompt=request.prompt,
        response=request.response,
        rating=request.rating,
        correction=request.correction,
        tags=request.tags,
    )

    return {
        "success": True,
        "feedback_id": feedback_id,
        "message": "Thanks! The agent will learn from this feedback.",
    }


@router.get("/quality")
async def get_quality(agent_id: str = Query(default="")):
    """Get quality metrics for one or all agents."""
    from app.services.intelligence.agent_learning import get_agent_learning
    learning = get_agent_learning()

    if agent_id:
        return learning.get_quality(agent_id)
    return {"agents": learning.get_all_quality()}


@router.get("/quality/alerts")
async def get_quality_alerts():
    """Get agents with declining quality (admin alert)."""
    from app.services.intelligence.agent_learning import get_agent_learning
    learning = get_agent_learning()
    return {"degrading_agents": learning.get_degrading_agents()}


# ─── Audit + Stats ──────────────────────────────────────────

@router.get("/audit")
async def get_audit_log(
    tenant_id: str = Query(...),
    limit: int = Query(default=50, le=500),
    event_type: str = Query(default=""),
):
    """Get audit log for a tenant."""
    from app.services.security.audit_logger import get_audit_logger
    audit = get_audit_logger()
    return {
        "logs": audit.get_tenant_log(tenant_id, limit, event_type or None),
        "stats": audit.get_stats(tenant_id),
    }


@router.get("/audit/security")
async def get_security_events(
    tenant_id: str = Query(default=""),
    limit: int = Query(default=50, le=200),
):
    """Get recent security events."""
    from app.services.security.audit_logger import get_audit_logger
    audit = get_audit_logger()
    return {"events": audit.get_security_events(tenant_id or None, limit)}


@router.get("/cache/stats")
async def get_cache_stats():
    """Get cache performance stats."""
    from app.services.optimization.response_cache import get_response_cache
    cache = get_response_cache()
    return cache.stats()
