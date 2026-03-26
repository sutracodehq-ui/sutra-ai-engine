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


def get_voice_config() -> dict:
    global _voice_config
    if _voice_config is None:
        # Load from main intelligence_config.yaml first
        from app.services.intelligence.brain import _cfg
        try:
            _voice_config = _cfg("voice", default={})
            if not _voice_config:
                # Fallback to legacy path if not in main yaml
                config_path = Path(__file__).resolve().parent.parent.parent / "config" / "voice.yaml"
                with open(config_path, "r") as f:
                    _voice_config = yaml.safe_load(f)
        except Exception:
            logger.warning("Voice config not found, using defaults")
            _voice_config = {
                "stt": {"provider": "openai", "model": "whisper-1"},
                "tts": {"provider": "openai", "model": "tts-1", "default_voice": "nova"},
                "max_verbatim_chars": 350
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
    config = get_voice_config()
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


def get_presigned_url(object_key: str, expires_in: int = 3600) -> str:
    """Generate a temporary signed URL for an R2 object."""
    if not object_key:
        return ""

    settings = get_settings()
    if not settings.r2_access_key or not settings.r2_endpoint:
        return ""

    try:
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

        url = s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": settings.r2_bucket, "Key": object_key},
            ExpiresIn=expires_in,
        )
        return url
    except Exception as e:
        logger.error(f"Failed to generate presigned URL: {e}")
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
    config = get_voice_config()
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


async def edge_tts_generate(text: str, voice: Optional[str] = None) -> Optional[bytes]:
    """
    Free TTS via Microsoft Edge's speech service.
    No API key needed. Supports 400+ voices including Hindi, Tamil, etc.
    Voice map is configured in intelligence_config.yaml under voice.edge.voice_map.
    """
    try:
        import edge_tts
        import io

        config = get_voice_config()
        edge_config = config.get("edge", {})
        default_edge_voice = edge_config.get("default_edge_voice", "en-IN-NeerjaNeural")
        
        # 1. Detect if text contains Hindi (Devnagari) characters
        has_hindi = any('\u0900' <= char <= '\u097F' for char in text)
        
        # 2. Determine voice name
        requested_voice = voice or config.get("default_voice", "nova")
        
        # 3. Auto-switch to Hindi voice if Devnagari detected
        if has_hindi:
            if requested_voice in ["nova", "shimmer", "fable"]:
                edge_voice = voice_map.get("swara") # Hindi Female
            else:
                edge_voice = voice_map.get("madhur") # Hindi Male
        else:
            edge_voice = voice_map.get(requested_voice, default_edge_voice)
            
        logger.info(f"🎤 [Edge-TTS] Requested: {requested_voice}, Final Selection: {edge_voice}, Has Hindi: {has_hindi}")

        # Get granular voice settings
        rate = edge_config.get("rate", "+0%")
        pitch = edge_config.get("pitch", "+0Hz")
        volume = edge_config.get("volume", "+0%")
        
        communicate = edge_tts.Communicate(
            text, 
            edge_voice,
            rate=rate,
            pitch=pitch,
            volume=volume
        )

        audio_buffer = io.BytesIO()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_buffer.write(chunk["data"])

        audio_bytes = audio_buffer.getvalue()
        if len(audio_bytes) > 0:
            logger.info(f"Edge TTS success: {len(audio_bytes)} bytes, voice={edge_voice}, rate={rate}, pitch={pitch}")
            return audio_bytes

        logger.warning("Edge TTS returned empty audio")
        return None
    except Exception as e:
        logger.warning(f"Edge TTS error: {e}")
        return None


async def text_to_speech(
    text: str,
    voice: Optional[str] = None,
) -> bytes:
    """
    Convert text to speech.
    Chain: Edge-TTS (free) → OpenAI (premium) → Sarvam (Indian).
    Raises HTTPException(503) if ALL providers fail — never unhandled 500.
    """
    from fastapi import HTTPException

    settings = get_settings()
    config = get_voice_config()
    tts_config = config.get("tts", {})
    errors = []

    # 1. Edge-TTS (FREE — no API key needed, always available)
    result = await edge_tts_generate(text, voice)
    if result:
        return result
    errors.append("Edge TTS: unavailable")

    # 2. OpenAI TTS (premium, expressive)
    try:
        api_key = settings.openai_api_key
        if not api_key:
            raise ValueError("OpenAI API key missing")

        voice_name = voice or tts_config.get("default_voice", "nova")

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
                    "voice": voice_name,
                    "response_format": tts_config.get("response_format", "mp3"),
                    "speed": tts_config.get("speed", 1.0),
                },
            )

            if response.status_code == 200:
                return response.content

            err_msg = f"OpenAI TTS ({response.status_code}): {response.text[:200]}"
            errors.append(err_msg)
            logger.warning(f"{err_msg}. Trying Sarvam fallback...")

    except Exception as e:
        errors.append(f"OpenAI TTS: {e}")
        logger.warning(f"OpenAI TTS error: {e}. Attempting Sarvam fallback...")

    # 3. Sarvam AI (Indian languages)
    try:
        result = await sarvam_tts(text)
        if result:
            return result
        errors.append("Sarvam TTS: returned empty audio")
    except Exception as e:
        errors.append(f"Sarvam TTS: {e}")
        logger.error(f"Sarvam TTS fallback also failed: {e}")

    # 4. All providers failed — return 503 (not 500)
    error_detail = " | ".join(errors)
    logger.error(f"All TTS providers failed: {error_detail}")
    raise HTTPException(
        status_code=503,
        detail=f"All TTS providers unavailable. Errors: {error_detail}"
    )


async def sarvam_tts(text: str) -> Optional[bytes]:
    """
    Convert text to speech using Sarvam AI (Bulbul v3).
    Excellent for Indian accents and regional languages.
    Returns None on failure (caller handles fallback).
    """
    settings = get_settings()
    api_key = settings.sarvam_api_key
    if not api_key:
        logger.warning("Sarvam API key not configured, skipping")
        return None

    try:
        import base64
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                "https://api.sarvam.ai/text-to-speech",
                headers={
                    "api-subscription-key": api_key,
                    "Content-Type": "application/json",
                },
                json={
                    "inputs": [text],
                    "target_language_code": "en-IN",
                    "speaker": "aditya",
                    "model": "bulbul:v3",
                },
            )

            if response.status_code != 200:
                logger.error(f"Sarvam TTS failed ({response.status_code}): {response.text[:200]}")
                return None

            result = response.json()
            audio_base64 = result.get("audio_content") or (result.get("audios", [None])[0] if result.get("audios") else None)

            if not audio_base64:
                logger.error(f"Sarvam TTS returned unexpected JSON: {result}")
                return None

            return base64.b64decode(audio_base64)
    except Exception as e:
        logger.error(f"Sarvam TTS error: {e}")
        return None

import re


def clean_for_tts(text: str) -> str:
    """
    Aggressive cleaning to strip ALL markdown, emojis, and formatting noise.
    Ensures the TTS only receives pure pronounceable text.
    """
    # 1. Remove code blocks (```...```) entirely
    text = re.sub(r'```[\s\S]*?```', '', text)
    
    # 2. Remove inline code (`...`) and keep the content
    text = re.sub(r'`([^`]+)`', r'\1', text)
    
    # 3. Remove ALL # characters (headers) - aggressive pass
    text = text.replace('#', '')
    
    # 4. Remove bold/italic markers (**, *, __, _)
    text = re.sub(r'[*_]{1,3}', '', text)
    
    # 5. Remove bullet points (- , * , • ) and blockquotes (>) at start of lines
    text = re.sub(r'^\s*[-*•>]\s+', '', text, flags=re.MULTILINE)
    
    # 6. Remove numbered lists (1. item)
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
    
    # 7. Remove links [text](url) → KEEP text, DROP url
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    
    # 8. Remove non-ASCII characters BUT KEEP Hindi (Devnagari range)
    # Range: \u0900-\u097F
    # This kills emojis, icons, and symbols while sparing English and Hindi text.
    text = re.sub(r'[^\x00-\x7F\u0900-\u097F]+', ' ', text)
    
    # 9. Collapse multiple newlines/spaces
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


async def simplify_for_voice(text: str, tone_override: Optional[str] = None) -> str:
    """
    Two-stage text cleaning for natural TTS:
    1. Regex strip: Remove markdown, emojis, formatting (instant)
    2. AI narration: Simplify long text via local Ollama (if enabled)
    """
    # Stage 1: Always strip markdown/emojis (instant, no AI)
    text = clean_for_tts(text)

    config = get_voice_config()
    narrator = config.get("narrator", {})
    if not narrator.get("enabled", True):
        return text

    # Stage 2: AI narration for long text only
    threshold = config.get("max_verbatim_chars", 350)
    if not tone_override and len(text) <= threshold:
        return text

    try:
        from app.services.intelligence.brain import get_brain
        brain = get_brain()

        prompt_tmpl = narrator.get("prompt", "Simplify for voice: {text}")
        
        # Inject tone override into the prompt if provided
        if tone_override:
            prompt = f"Convert this text to a {tone_override} tone for voice output. STRICTLY preserve the original language and script (Hindi, Maithili, etc.). Do not translate to English: {text}"
        else:
            prompt = prompt_tmpl.format(text=text)

        response = await brain.execute(
            prompt=prompt,
            system_prompt="You are a professional, helpful, and sophisticated AI assistant.",
            agent_type="summarizer",
            model=narrator.get("model", "qwen2.5:3b")
        )

        natural_text = response.content.strip()
        # Clean the AI output too (it might add markdown)
        natural_text = clean_for_tts(natural_text)
        logger.info(f"Voice narrator: {len(text)} → {len(natural_text)} chars")
        return natural_text
    except Exception as e:
        logger.warning(f"Voice simplification failed: {e}")
        return text[:threshold] + "..."


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
            # Natural Narrator: Simplify text for human-like speech
            speech_text = await simplify_for_voice(agent_response.content)
            
            tts_bytes = await text_to_speech(speech_text, voice)
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
        "audio_url": get_presigned_url(r2_key),
        "voice_response_r2_key": voice_response_key,
        "voice_response_url": get_presigned_url(voice_response_key),
    }
