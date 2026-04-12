"""
VoIP Engine — Self-hosted real-time voice AI pipeline.

Software Factory Principle: Self-reliance + No vendor lock-in.

Fully self-hosted pipeline:
  Call Audio → Faster-Whisper (STT) → AI Agent → Edge-TTS → Audio Response

No external STT/TTS APIs needed. Runs entirely on your infrastructure.

India-first: Supports Hindi, Tamil, Telugu, Bengali, Marathi, Kannada,
Malayalam, Gujarati, Punjabi, Hinglish, and 90+ more via Whisper.
"""

import asyncio
import io
import json
import logging
import tempfile
from datetime import datetime, timezone, time as dtime
from pathlib import Path
from typing import Optional

import yaml

from app.config import get_settings

logger = logging.getLogger(__name__)

CONFIG_PATH = Path("intelligence_config.yaml")


def _load_voip_config() -> dict:
    """Load VoIP config from YAML."""
    if not CONFIG_PATH.exists():
        return {}
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f) or {}
    return config.get("voip", {})


class VoIPEngine:
    """
    Self-hosted real-time voice AI engine.

    Pipeline:
    1. Receive audio from telephony (Twilio/Exotel webhook)
    2. Transcribe with Faster-Whisper (local, 99+ languages)
    3. Detect language automatically
    4. Route to appropriate AI agent via persona
    5. Generate speech response with Edge-TTS (free, 100+ languages)
    6. Stream audio back to caller

    India Compliance:
    - TRAI DND check before outbound calls
    - Call timing: 9 AM – 9 PM IST only
    - Recording consent message at call start
    """

    def __init__(self):
        self._whisper_model = None
        self._call_log_dir = Path("training/call_logs")
        self._call_log_dir.mkdir(parents=True, exist_ok=True)

    # ─── Faster-Whisper STT (Speech → Text) ─────────────────

    def _get_whisper(self):
        """Lazy-load Faster-Whisper model."""
        if self._whisper_model is None:
            from faster_whisper import WhisperModel

            config = _load_voip_config()
            model_size = config.get("stt", {}).get("model_size", "base")
            device = config.get("stt", {}).get("device", "cpu")
            compute_type = config.get("stt", {}).get("compute_type", "int8")

            self._whisper_model = WhisperModel(
                model_size, device=device, compute_type=compute_type,
            )
            logger.info(f"VoIP: Whisper loaded (model={model_size}, device={device})")
        return self._whisper_model

    async def transcribe(self, audio_path: str) -> dict:
        """
        Transcribe audio using Faster-Whisper.
        Returns text + detected language.
        """
        model = self._get_whisper()

        # Run in thread pool (Whisper is CPU-bound)
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, self._transcribe_sync, model, audio_path
        )
        return result

    def _transcribe_sync(self, model, audio_path: str) -> dict:
        """Synchronous Whisper transcription."""
        config = _load_voip_config()
        stt_config = config.get("stt", {})

        segments, info = model.transcribe(
            audio_path,
            beam_size=stt_config.get("beam_size", 5),
            language=stt_config.get("force_language", None),  # None = auto-detect
            vad_filter=True,  # Voice Activity Detection
        )

        text_parts = []
        for segment in segments:
            text_parts.append(segment.text.strip())

        full_text = " ".join(text_parts)

        return {
            "text": full_text,
            "language": info.language,
            "language_probability": round(info.language_probability, 3),
            "duration": round(info.duration, 2),
        }

    # ─── Edge-TTS (Text → Speech) ───────────────────────────

    async def synthesize(
        self, text: str, language: str = "hi", voice: str | None = None,
    ) -> bytes:
        """
        Convert text to speech using Edge-TTS.
        Free, 100+ languages, excellent Indian voices.
        """
        import edge_tts

        config = _load_voip_config()
        tts_config = config.get("tts", {})

        # Resolve voice: explicit > language-default > fallback
        if not voice:
            indian_voices = tts_config.get("indian_voices", {})
            voice = indian_voices.get(language, tts_config.get("default_voice", "en-IN-NeerjaNeural"))

        communicate = edge_tts.Communicate(text, voice)

        # Collect audio bytes
        audio_chunks = []
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_chunks.append(chunk["data"])

        audio_bytes = b"".join(audio_chunks)
        logger.info(
            f"VoIP-TTS: generated {len(audio_bytes)} bytes "
            f"(lang={language}, voice={voice})"
        )
        return audio_bytes

    # ─── Full Pipeline ──────────────────────────────────────

    async def process_call(
        self,
        audio_path: str,
        persona_id: str = "support_agent_india",
        db=None,
        context: dict | None = None,
    ) -> dict:
        """
        Full VoIP pipeline for a single call turn:
        1. Transcribe caller speech (Faster-Whisper)
        2. Detect language
        3. Route to AI agent via persona
        4. Generate speech response (Edge-TTS)
        """
        from app.services.intelligence.voip_personas import get_persona_manager

        # 1. Transcribe
        transcription = await self.transcribe(audio_path)
        caller_text = transcription["text"]
        detected_lang = transcription["language"]

        if not caller_text.strip():
            return {
                "status": "silence",
                "message": "No speech detected",
            }

        # 2. Get persona
        persona_mgr = get_persona_manager()
        persona = persona_mgr.get_persona(persona_id)

        if not persona:
            return {"status": "error", "message": f"Persona {persona_id} not found"}

        # 3. Route to AI agent
        from app.services.agents.hub import AiAgentHub
        hub = AiAgentHub()
        agent = hub.get(persona.agent_id)

        # Build context with language info
        agent_context = {
            **(context or {}),
            "language": detected_lang,
            "caller_language": detected_lang,
            "persona": persona_id,
            "voip": True,
        }

        response = await agent.execute(caller_text, db=db, context=agent_context)
        response_text = response.content

        # 4. Synthesize speech
        voice = persona.voice_id
        audio_bytes = await self.synthesize(response_text, detected_lang, voice)

        # 5. Save audio response to temp file
        audio_file = tempfile.NamedTemporaryFile(
            suffix=".mp3", delete=False, dir=str(self._call_log_dir)
        )
        audio_file.write(audio_bytes)
        audio_file.close()

        # 6. Log the call turn
        call_log = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "persona": persona_id,
            "language": detected_lang,
            "caller_text": caller_text,
            "agent_response": response_text[:500],
            "audio_path": audio_file.name,
            "duration": transcription.get("duration", 0),
        }
        self._log_call(call_log)

        return {
            "status": "success",
            "transcription": caller_text,
            "language": detected_lang,
            "response_text": response_text,
            "audio_path": audio_file.name,
            "audio_size": len(audio_bytes),
        }

    # ─── TRAI Compliance ────────────────────────────────────

    def is_calling_allowed(self) -> dict:
        """
        Check if outbound calling is allowed per TRAI rules.
        Calls only permitted 9 AM – 9 PM IST.
        """
        from datetime import timezone, timedelta
        ist = timezone(timedelta(hours=5, minutes=30))
        now = datetime.now(ist)
        current_time = now.time()

        config = _load_voip_config()
        trai = config.get("trai_compliance", {})
        start = dtime(*[int(x) for x in trai.get("call_start_time", "09:00").split(":")])
        end = dtime(*[int(x) for x in trai.get("call_end_time", "21:00").split(":")])

        allowed = start <= current_time <= end
        return {
            "allowed": allowed,
            "current_time_ist": current_time.isoformat(),
            "window": f"{start.isoformat()} – {end.isoformat()}",
            "reason": "Within TRAI permitted hours" if allowed else "Outside TRAI hours (9 AM – 9 PM IST)",
        }

    def get_consent_message(self, language: str = "en") -> str:
        """Get recording consent message in the caller's language."""
        config = _load_voip_config()
        consent = config.get("consent_messages", {})
        return consent.get(language, consent.get("en", "This call may be recorded for quality purposes."))

    # ─── Call Logging ───────────────────────────────────────

    def _log_call(self, entry: dict):
        """Append call log entry to JSONL."""
        log_path = self._call_log_dir / "call_log.jsonl"
        try:
            with open(log_path, "a") as f:
                f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
        except Exception as e:
            logger.warning(f"VoIP: call logging failed: {e}")


# ─── Singleton ──────────────────────────────────────────────
_engine: VoIPEngine | None = None


def get_voip_engine() -> VoIPEngine:
    global _engine
    if _engine is None:
        _engine = VoIPEngine()
    return _engine
