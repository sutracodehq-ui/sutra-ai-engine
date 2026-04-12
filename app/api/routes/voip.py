"""
VoIP API Routes — FastAPI endpoints for voice calls.

POST /v1/voip/call      — Process a call turn (audio in → audio out)
POST /v1/voip/outbound  — Initiate outbound call
GET  /v1/voip/personas  — List available personas
GET  /v1/voip/calls     — Call history
GET  /v1/voip/compliance — Check TRAI compliance status
"""

import logging
import tempfile
from typing import Optional

from fastapi import APIRouter, Depends, File, UploadFile, Query
from pydantic import BaseModel, Field

from sqlalchemy.ext.asyncio import AsyncSession
from app.db.session import get_db

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1/voip", tags=["voip"])


class CallResponse(BaseModel):
    status: str
    transcription: Optional[str] = None
    language: Optional[str] = None
    response_text: Optional[str] = None
    audio_path: Optional[str] = None


class OutboundRequest(BaseModel):
    phone_number: str = Field(..., description="+91XXXXXXXXXX format")
    persona: str = Field(default="cold_caller_india")
    message: str = Field(..., description="What to say on the call")


@router.post("/call", response_model=CallResponse)
async def process_call(
    audio: UploadFile = File(...),
    persona: str = Query(default="support_agent_india"),
    db: AsyncSession = Depends(get_db),
):
    """
    Process a single call turn.
    Upload caller audio → get AI response audio.
    """
    from app.services.intelligence.voip_engine import get_voip_engine
    engine = get_voip_engine()

    # Save uploaded audio to temp file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        content = await audio.read()
        tmp.write(content)
        tmp_path = tmp.name

    result = await engine.process_call(
        audio_path=tmp_path,
        persona_id=persona,
        db=db,
    )

    return CallResponse(**result)


@router.post("/outbound")
async def initiate_outbound(request: OutboundRequest):
    """
    Initiate an outbound call.
    Checks TRAI compliance before calling.
    """
    from app.services.intelligence.voip_engine import get_voip_engine
    engine = get_voip_engine()

    # TRAI compliance check
    compliance = engine.is_calling_allowed()
    if not compliance["allowed"]:
        return {
            "status": "blocked",
            "reason": compliance["reason"],
            "window": compliance["window"],
        }

    # Synthesize the outbound message
    from app.services.intelligence.voip_personas import get_persona_manager
    mgr = get_persona_manager()
    persona = mgr.get_persona(request.persona)

    if not persona:
        return {"status": "error", "message": f"Persona {request.persona} not found"}

    audio_bytes = await engine.synthesize(
        text=request.message,
        language="en",
        voice=persona.voice_id,
    )

    return {
        "status": "ready",
        "phone": request.phone_number,
        "persona": request.persona,
        "audio_size": len(audio_bytes),
        "message": "Audio generated. Connect to telephony provider to initiate call.",
    }


@router.get("/personas")
async def list_personas():
    """List all available VoIP personas."""
    from app.services.intelligence.voip_personas import get_persona_manager
    mgr = get_persona_manager()
    return mgr.list_personas()


@router.get("/compliance")
async def check_compliance():
    """Check TRAI compliance status for outbound calls."""
    from app.services.intelligence.voip_engine import get_voip_engine
    engine = get_voip_engine()
    return engine.is_calling_allowed()
