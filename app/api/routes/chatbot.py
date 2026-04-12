"""
Chatbot API Routes — Endpoints for embeddable chatbot brain.

POST /v1/chatbot/chat         — Text chat message
POST /v1/chatbot/voice        — Voice message (WebRTC audio)
POST /v1/chatbot/webhook      — WhatsApp reply webhook (owner answers)
POST /v1/chatbot/knowledge    — Import FAQ into brand knowledge base
GET  /v1/chatbot/sessions     — List active sessions
GET  /v1/chatbot/escalations  — List pending escalations
GET  /v1/chatbot/knowledge/stats — Brand knowledge stats
"""

import logging
import tempfile
from typing import Optional

from fastapi import APIRouter, Depends, File, UploadFile, Query
from pydantic import BaseModel, Field

from sqlalchemy.ext.asyncio import AsyncSession
from app.db.session import get_db

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1/chatbot", tags=["chatbot"])


# ─── Request/Response Models ────────────────────────────────

class ChatRequest(BaseModel):
    session_id: str = Field(..., description="Unique session identifier")
    brand_id: str = Field(..., description="Brand identifier")
    message: str = Field(..., description="Customer message")
    visitor_id: Optional[str] = None
    language: Optional[str] = None
    channel: str = Field(default="text", description="text, voice, or webrtc")


class ChatResponse(BaseModel):
    session_id: str
    response: str
    confidence: float
    escalated: bool
    language: Optional[str] = None
    channel: str = "text"


class WebhookRequest(BaseModel):
    escalation_id: str = Field(..., description="Reference from WhatsApp msg")
    answer: str = Field(..., description="Brand owner's answer")


class FAQImportRequest(BaseModel):
    brand_id: str
    items: list[dict] = Field(..., description='[{"question": "...", "answer": "..."}]')


# ─── Endpoints ──────────────────────────────────────────────

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, db: AsyncSession = Depends(get_db)):
    """
    Process a text chat message from a customer.
    Returns AI response + whether it was escalated to brand owner.
    """
    from app.services.intelligence.chatbot_engine import get_chatbot_engine
    engine = get_chatbot_engine()

    result = await engine.chat(
        session_id=request.session_id,
        brand_id=request.brand_id,
        message=request.message,
        channel=request.channel,
        visitor_id=request.visitor_id,
        language=request.language,
        db=db,
    )

    return ChatResponse(**result)


@router.post("/voice")
async def voice_chat(
    audio: UploadFile = File(...),
    session_id: str = Query(...),
    brand_id: str = Query(...),
    db: AsyncSession = Depends(get_db),
):
    """
    Process a voice message (from WebRTC).
    Transcribes → chats → returns response audio path.
    """
    from app.services.intelligence.chatbot_engine import get_chatbot_engine
    engine = get_chatbot_engine()

    # Save uploaded audio to temp file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        content = await audio.read()
        tmp.write(content)
        tmp_path = tmp.name

    result = await engine.voice_chat(
        session_id=session_id,
        brand_id=brand_id,
        audio_path=tmp_path,
        db=db,
    )

    return result


@router.post("/webhook")
async def whatsapp_webhook(request: WebhookRequest):
    """
    WhatsApp reply webhook.
    When brand owner replies to an escalation, this endpoint
    receives their answer, teaches the AI, and resolves the escalation.
    Also pushes the answer to customer's WebSocket in real-time.
    """
    from app.services.intelligence.brain import get_brain
    brain = get_brain()
    result = await brain.resolve_escalation(request.escalation_id, request.answer)

    # Push answer to customer's WebSocket in real-time
    if result.get("session_id") and result.get("status") == "resolved":
        try:
            from app.api.routes.chatbot_ws import push_escalation_resolved
            await push_escalation_resolved(result["session_id"], request.answer)
        except Exception:
            pass  # WS may not be connected

    return result


@router.post("/knowledge")
async def import_knowledge(request: FAQImportRequest):
    """Bulk import FAQ items into a brand's knowledge base."""
    from app.services.intelligence.memory import get_memory
    mem = get_memory()

    result = await mem.brand_import_faq(
        brand_id=request.brand_id,
        items=request.items,
    )

    return result


@router.get("/escalations")
async def list_escalations(brand_id: Optional[str] = None):
    """List pending escalations (queries waiting for brand owner answer)."""
    from app.services.intelligence.brain import get_brain
    brain = get_brain()
    return brain.get_pending_escalations(brand_id)


@router.get("/knowledge/stats")
async def knowledge_stats(brand_id: str = Query(...)):
    """Get knowledge base stats for a brand."""
    from app.services.intelligence.memory import get_memory
    mem = get_memory()
    # Stats: count documents in brand collection
    try:
        chroma = mem._get_chroma() if hasattr(mem, '_get_chroma') else None
        from app.services.intelligence.memory import _get_chroma
        client = _get_chroma()
        if client:
            coll = client.get_or_create_collection(name=f"brand_{brand_id}_knowledge")
            return {"brand_id": brand_id, "total_entries": coll.count()}
    except Exception:
        pass
    return {"brand_id": brand_id, "total_entries": 0}
