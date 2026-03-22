"""
Voice API — Accept voice input, transcribe, process, and respond.

Full pipeline:
  Audio Upload → R2 Storage → Whisper STT → Language Auto-Detect → Agent → Response
  Optionally: → TTS Voice Response → R2 Storage
"""

from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException
from typing import Optional

from app.dependencies import DbSession, get_current_tenant
from app.models.tenant import Tenant
from app.services.voice.voice_service import (
    process_voice_input,
    transcribe_audio,
    text_to_speech,
    upload_to_r2,
)

router = APIRouter(prefix="/voice", tags=["voice"])


@router.post("/process", summary="Process Voice Input")
async def process_voice(
    file: UploadFile = File(..., description="Audio file (webm, mp3, wav, ogg, flac, m4a)"),
    agent_type: str = Form(default="copywriter", description="Agent to process the voice input"),
    generate_voice_reply: bool = Form(default=False, description="Generate TTS audio reply"),
    voice: Optional[str] = Form(default=None, description="TTS voice: alloy, echo, fable, onyx, nova, shimmer"),
    tenant: Tenant = Depends(get_current_tenant),
    db: DbSession = None,
):
    """
    Accept a voice message, auto-detect the language, and process it through any agent.

    **Full Pipeline:**
    1. 🎤 Upload audio to R2 (`voice/{tenant}/{date}/{id}.webm`)
    2. 📝 Transcribe using OpenAI Whisper (auto-detects all Indian languages)
    3. 🌐 Detect language (Hindi, Maithili, Bhojpuri, Tamil, etc.)
    4. 🤖 Route transcription to the specified agent with language context
    5. 🔊 Optionally generate a voice response via TTS

    **Supported Formats:** webm, mp3, wav, ogg, flac, m4a (up to 25MB)

    **Auto Language Detection:** Whisper natively detects 99+ languages including:
    Hindi, Bengali, Tamil, Telugu, Marathi, Gujarati, Kannada, Malayalam,
    Punjabi, Odia, Assamese, Maithili, Bhojpuri, and more.

    **No toggle needed** — the engine detects the language and responds in it automatically.

    **Example:**
    - User sends voice in Bhojpuri → Agent detects Bhojpuri → Responds in Bhojpuri
    - User sends voice in Tamil → Agent detects Tamil → Responds in Tamil
    """
    # Validate file type
    allowed_types = [
        "audio/webm", "audio/mp3", "audio/mpeg", "audio/wav",
        "audio/ogg", "audio/flac", "audio/m4a", "audio/mp4",
        "audio/x-m4a", "video/webm",  # Some browsers send webm as video
    ]
    if file.content_type and file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio format: {file.content_type}. Use webm, mp3, wav, ogg, flac, or m4a.",
        )

    # Read file bytes
    file_bytes = await file.read()

    # Validate file size (25MB max for Whisper)
    max_size = 25 * 1024 * 1024
    if len(file_bytes) > max_size:
        raise HTTPException(
            status_code=400,
            detail=f"Audio file too large ({len(file_bytes) / 1024 / 1024:.1f}MB). Maximum is 25MB.",
        )

    result = await process_voice_input(
        file_bytes=file_bytes,
        filename=file.filename or "voice.webm",
        tenant_slug=tenant.slug,
        content_type=file.content_type or "audio/webm",
        agent_type=agent_type,
        generate_voice_reply=generate_voice_reply,
        voice=voice,
        db=db,
    )

    if "error" in result:
        raise HTTPException(status_code=422, detail=result["error"])

    return {
        **result,
        "detected_language": result.get("detected_language", "unknown"),
        "agent_type": agent_type,
        "voice_reply_generated": generate_voice_reply,
    }


@router.post("/transcribe", summary="Transcribe Audio Only")
async def transcribe_only(
    file: UploadFile = File(..., description="Audio file to transcribe"),
    tenant: Tenant = Depends(get_current_tenant),
):
    """
    Transcribe audio without processing through an agent.

    **Returns:**
    - `text`: Transcribed text
    - `language`: Auto-detected language code
    - `duration`: Audio duration in seconds

    Useful for voice-to-text features in the UI.
    """
    file_bytes = await file.read()

    # Upload to R2 for archival
    r2_key = await upload_to_r2(
        file_bytes,
        f"transcribe_{file.filename}",
        tenant.slug,
        file.content_type or "audio/webm",
    )

    transcription = await transcribe_audio(file_bytes, file.filename or "audio.webm")

    return {
        "text": transcription["text"],
        "language": transcription["language"],
        "duration": transcription["duration"],
        "r2_key": r2_key,
    }


@router.post("/speak", summary="Text-to-Speech")
async def speak(
    text: str = Form(..., description="Text to convert to speech"),
    voice: Optional[str] = Form(default="alloy", description="Voice: alloy, echo, fable, onyx, nova, shimmer"),
    tenant: Tenant = Depends(get_current_tenant),
):
    """
    Convert text to speech (TTS).

    **Returns:** Audio file URL stored in R2.

    **Voices:** alloy, echo, fable, onyx, nova, shimmer
    """
    audio_bytes = await text_to_speech(text, voice)

    import uuid
    filename = f"tts_{uuid.uuid4().hex[:8]}.mp3"
    r2_key = await upload_to_r2(audio_bytes, filename, tenant.slug, "audio/mpeg")

    return {
        "r2_key": r2_key,
        "voice": voice,
        "text_length": len(text),
    }
