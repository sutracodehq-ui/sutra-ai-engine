"""
Voice Service — Full voice pipeline.

Pipeline: Audio Upload → R2 Storage → Whisper STT → Language Detection → Agent → TTS Response

Config-driven: all settings loaded from config/voice.yaml.
"""

import io
import uuid
import logging
import yaml
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import httpx

from app.config import get_settings

logger = logging.getLogger(__name__)

# ─── Config Loader ───────────────────────────────────────────

_voice_config: dict | None = None


def _load_voice_config() -> dict:
    global _voice_config
    if _voice_config is None:
        config_path = Path(__file__).resolve().parent.parent.parent / "config" / "voice.yaml"
        try:
            with open(config_path, "r") as f:
                _voice_config = yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning("Voice config not found, using defaults")
            _voice_config = {
                "stt": {"provider": "openai", "model": "whisper-1"},
                "tts": {"provider": "openai", "model": "tts-1", "default_voice": "alloy"},
            }
    return _voice_config


# ─── R2 Storage ──────────────────────────────────────────────

async def upload_to_r2(
    file_bytes: bytes,
    filename: str,
    tenant_slug: str,
    content_type: str = "audio/webm",
) -> str:
    """Upload audio file to Cloudflare R2. Returns the R2 object key."""
    settings = get_settings()

    if not settings.r2_access_key or not settings.r2_endpoint:
        logger.warning("R2 not configured, skipping voice upload")
        return ""

    date_prefix = datetime.now(timezone.utc).strftime("%Y/%m/%d")
    config = _load_voice_config()
    bucket_prefix = config.get("storage", {}).get("bucket_prefix", "voice")
    object_key = f"{bucket_prefix}/{tenant_slug}/{date_prefix}/{filename}"

    try:
        # Use boto3-compatible S3 API for R2
        import boto3
        from botocore.config import Config as BotoConfig

        s3 = boto3.client(
            "s3",
            endpoint_url=settings.r2_endpoint,
            aws_access_key_id=settings.r2_access_key,
            aws_secret_access_key=settings.r2_secret_key,
            config=BotoConfig(signature_version="s3v4"),
            region_name="auto",
        )

        s3.put_object(
            Bucket=settings.r2_bucket,
            Key=object_key,
            Body=file_bytes,
            ContentType=content_type,
        )

        logger.info(f"🎤 Uploaded voice recording to R2: {object_key}")
        return object_key

    except Exception as e:
        logger.error(f"R2 upload failed: {e}")
        return ""


# ─── Speech-to-Text (Transcription) ─────────────────────────

async def transcribe_audio(
    file_bytes: bytes,
    filename: str,
    language_hint: Optional[str] = None,
) -> dict:
    """
    Transcribe audio using OpenAI Whisper API.

    Returns:
        {
            "text": "transcribed text",
            "language": "detected language code (e.g., hi, mai, en)",
            "duration": 12.5
        }
    """
    settings = get_settings()
    config = _load_voice_config()
    stt_config = config.get("stt", {})

    api_key = settings.openai_api_key
    if not api_key:
        raise ValueError("OpenAI API key not configured for Whisper STT")

    # Build multipart form data for Whisper API
    form_data = {
        "model": (None, stt_config.get("model", "whisper-1")),
        "response_format": (None, "verbose_json"),  # Get language + duration
    }

    # Add language hint if provided (helps accuracy but doesn't restrict)
    if language_hint and language_hint != "auto":
        form_data["language"] = (None, language_hint)

    files = {
        "file": (filename, file_bytes, "audio/webm"),
    }

    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(
            "https://api.openai.com/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {api_key}"},
            data={k: v[1] for k, v in form_data.items()},
            files=files,
        )

        if response.status_code != 200:
            logger.error(f"Whisper API error: {response.status_code} - {response.text}")
            raise ValueError(f"Transcription failed: {response.text}")

        result = response.json()
        return {
            "text": result.get("text", ""),
            "language": result.get("language", "unknown"),
            "duration": result.get("duration", 0),
        }


# ─── Text-to-Speech (Voice Response) ────────────────────────

async def text_to_speech(
    text: str,
    voice: Optional[str] = None,
) -> bytes:
    """
    Convert text to speech using OpenAI TTS API.

    Returns audio bytes (MP3 format).
    """
    settings = get_settings()
    config = _load_voice_config()
    tts_config = config.get("tts", {})

    api_key = settings.openai_api_key
    if not api_key:
        raise ValueError("OpenAI API key not configured for TTS")

    voice = voice or tts_config.get("default_voice", "alloy")

    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(
            "https://api.openai.com/v1/audio/speech",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": tts_config.get("model", "tts-1"),
                "input": text,
                "voice": voice,
                "response_format": tts_config.get("response_format", "mp3"),
                "speed": tts_config.get("speed", 1.0),
            },
        )

        if response.status_code != 200:
            logger.error(f"TTS API error: {response.status_code} - {response.text}")
            raise ValueError(f"TTS failed: {response.text}")

        return response.content


# ─── Full Pipeline ───────────────────────────────────────────

async def process_voice_input(
    file_bytes: bytes,
    filename: str,
    tenant_slug: str,
    content_type: str = "audio/webm",
    agent_type: str = "copywriter",
    language_hint: Optional[str] = None,
    generate_voice_reply: bool = False,
    voice: Optional[str] = None,
    db=None,
) -> dict:
    """
    Full voice pipeline:
    1. Upload audio to R2
    2. Transcribe with Whisper (auto-detect language)
    3. Route to agent with detected language
    4. Optionally generate voice response via TTS

    Returns complete pipeline result.
    """
    file_id = str(uuid.uuid4())[:12]
    ext = filename.rsplit(".", 1)[-1] if "." in filename else "webm"
    r2_filename = f"{file_id}.{ext}"

    # Step 1: Upload to R2 (async, non-blocking)
    r2_key = await upload_to_r2(file_bytes, r2_filename, tenant_slug, content_type)

    # Step 2: Transcribe
    transcription = await transcribe_audio(file_bytes, filename, language_hint)
    detected_language = transcription["language"]
    transcribed_text = transcription["text"]

    if not transcribed_text.strip():
        return {
            "transcription": transcription,
            "r2_key": r2_key,
            "error": "Could not transcribe audio — no speech detected.",
        }

    # Step 3: Route to agent with language context
    from app.services.agents.hub import get_agent_hub
    hub = get_agent_hub()

    # Map Whisper language codes to our language codes
    context = {
        "tenant_slug": tenant_slug,
        "language": detected_language,
        "input_type": "voice",
    }

    agent_response = await hub.run(agent_type, transcribed_text, context=context, db=db)

    # Step 4: Optionally generate voice response
    voice_response_key = None
    if generate_voice_reply and agent_response.content:
        try:
            tts_bytes = await text_to_speech(agent_response.content, voice)
            tts_filename = f"{file_id}_reply.mp3"
            voice_response_key = await upload_to_r2(
                tts_bytes, tts_filename, tenant_slug, "audio/mpeg"
            )
        except Exception as e:
            logger.error(f"TTS generation failed: {e}")

    return {
        "transcription": transcription,
        "detected_language": detected_language,
        "agent_type": agent_type,
        "agent_response": agent_response.content,
        "r2_key": r2_key,
        "voice_response_r2_key": voice_response_key,
    }
