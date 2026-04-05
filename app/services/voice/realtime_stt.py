"""
Real-Time STT — Provider-abstracted speech-to-text for live audio.

Software Factory:
- Config-driven: provider selection from YAML → realtime.stt_provider
- Provider registry: dict[str, Type] — add new providers with zero if/else
- Language normalization: ISO-639-1 ↔ full names (shared utility)
- No hardcoded URLs, models, or thresholds

Pipeline: audio bytes → provider → {"text": "...", "language": "hi", "duration": 2.5}
"""

import logging
from abc import ABC, abstractmethod
from typing import Optional, Type

import httpx

from app.config import get_settings
from app.services.voice.config import get_voice_config

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
#  LANGUAGE NORMALIZATION
#  Groq returns full names ("english") but API needs ISO-639-1 ("en")
# ═══════════════════════════════════════════════════════════════

_LANG_TO_ISO: dict[str, str] = {
    "english": "en", "hindi": "hi", "hinglish": "hi",
    "spanish": "es", "french": "fr", "german": "de",
    "japanese": "ja", "chinese": "zh", "korean": "ko",
    "portuguese": "pt", "russian": "ru", "arabic": "ar",
    "italian": "it", "dutch": "nl", "turkish": "tr",
    "greek": "el", "tamil": "ta", "telugu": "te",
    "bengali": "bn", "marathi": "mr", "gujarati": "gu",
    "kannada": "kn", "malayalam": "ml", "punjabi": "pa",
    "urdu": "ur", "thai": "th", "vietnamese": "vi",
    "indonesian": "id", "malay": "ms", "swedish": "sv",
    "polish": "pl", "czech": "cs", "romanian": "ro",
}


def normalize_language(raw: str) -> str:
    """Convert full language name to ISO-639-1 code."""
    if not raw or raw == "unknown":
        return "unknown"
    lower = raw.lower().strip()
    return lower if len(lower) <= 3 else _LANG_TO_ISO.get(lower, lower)


# ═══════════════════════════════════════════════════════════════
#  CONTENT TYPE → FILE EXTENSION
# ═══════════════════════════════════════════════════════════════

_CONTENT_TYPE_TO_EXT: dict[str, str] = {
    "audio/webm": "webm", "audio/wav": "wav", "audio/mp3": "mp3",
    "audio/mpeg": "mp3",  "audio/ogg": "ogg", "audio/flac": "flac",
    "audio/mp4": "mp4",   "audio/m4a": "m4a",
}

# Minimum audio size (bytes) to be a valid container
_MIN_AUDIO_SIZE = 1000


def resolve_extension(content_type: str) -> tuple[str, str]:
    """Resolve (filename, base_type) from a content type. Strips codec suffix."""
    base = content_type.split(";")[0].strip()
    ext = _CONTENT_TYPE_TO_EXT.get(base, "webm")
    return f"chunk.{ext}", base


# ═══════════════════════════════════════════════════════════════
#  ABSTRACT BASE — All STT providers implement this
# ═══════════════════════════════════════════════════════════════

class RealtimeSTTProvider(ABC):
    """Abstract base for real-time STT providers."""

    @abstractmethod
    async def transcribe(
        self,
        audio_bytes: bytes,
        language_hint: Optional[str] = None,
        content_type: str = "audio/webm",
    ) -> dict:
        """Transcribe audio bytes → {"text": "...", "language": "hi", "duration": 2.5}"""
        ...


# ═══════════════════════════════════════════════════════════════
#  GROQ WHISPER — Primary provider (~300ms latency)
# ═══════════════════════════════════════════════════════════════

class GroqChunkedSTT(RealtimeSTTProvider):
    """Groq Whisper — blazing-fast REST-based STT."""

    def __init__(self):
        cfg = get_voice_config().get("realtime", {})
        self._cfg = cfg.get("groq", {})

    async def transcribe(
        self,
        audio_bytes: bytes,
        language_hint: Optional[str] = None,
        content_type: str = "audio/webm",
    ) -> dict:
        api_key = get_settings().groq_api_key
        if not api_key:
            raise ValueError("GROQ_API_KEY not configured")

        if len(audio_bytes) < _MIN_AUDIO_SIZE:
            return {"text": "", "language": "unknown", "duration": 0}

        filename, _ = resolve_extension(content_type)

        # Build form data from config (model, temperature, language, prompt — all YAML-driven)
        form = self._build_form(language_hint)
        files = {"file": (filename, audio_bytes, content_type)}

        # URL from providers config (never hardcode)
        from app.services.intelligence.brain import _cfg
        base_url = _cfg("providers", default={}).get("groq", {}).get(
            "base_url", "https://api.groq.com/openai/v1"
        )

        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"{base_url.rstrip('/')}/audio/transcriptions",
                headers={"Authorization": f"Bearer {api_key}"},
                data={k: v[1] for k, v in form.items()},
                files=files,
            )

        if resp.status_code != 200:
            logger.error(f"Groq STT {resp.status_code}: {resp.text[:200]}")
            raise ValueError(f"Groq STT failed ({resp.status_code}): {resp.text[:100]}")

        return self._parse_response(resp.json())

    def _build_form(self, language_hint: Optional[str]) -> dict:
        """Config-driven form params — no hardcoded values."""
        form = {
            "model": (None, self._cfg.get("model", "whisper-large-v3-turbo")),
            "response_format": (None, "verbose_json"),
            "temperature": (None, str(self._cfg.get("temperature", 0.0))),
        }
        hint = language_hint or self._cfg.get("language_hint")
        if hint:
            form["language"] = (None, hint)
        prompt = self._cfg.get("initial_prompt")
        if prompt:
            form["prompt"] = (None, prompt)
        return form

    @staticmethod
    def _parse_response(result: dict) -> dict:
        """Normalize Groq's response to standard format."""
        return {
            "text": result.get("text", "").strip(),
            "language": normalize_language(result.get("language", "unknown")),
            "duration": result.get("duration", 0),
        }


# ═══════════════════════════════════════════════════════════════
#  OPENAI WHISPER — Fallback provider
# ═══════════════════════════════════════════════════════════════

class OpenAIBatchSTT(RealtimeSTTProvider):
    """OpenAI Whisper — batch fallback STT via existing voice_service."""

    async def transcribe(
        self,
        audio_bytes: bytes,
        language_hint: Optional[str] = None,
        content_type: str = "audio/webm",
    ) -> dict:
        from app.services.voice.voice_service import transcribe_audio
        filename, _ = resolve_extension(content_type)
        return await transcribe_audio(audio_bytes, filename, language_hint)


# ═══════════════════════════════════════════════════════════════
#  FACTORY — Registry-based provider resolution (no if/elif)
# ═══════════════════════════════════════════════════════════════

_PROVIDER_REGISTRY: dict[str, Type[RealtimeSTTProvider]] = {
    "groq": GroqChunkedSTT,
    "openai": OpenAIBatchSTT,
    "openai_batch": OpenAIBatchSTT,
}

_provider_cache: dict[str, RealtimeSTTProvider] = {}


def get_realtime_stt(provider_override: Optional[str] = None) -> RealtimeSTTProvider:
    """Factory — returns STT provider from YAML config. Registry-based, no conditionals."""
    name = provider_override or get_voice_config().get("realtime", {}).get("stt_provider", "groq")

    if name not in _provider_cache:
        cls = _PROVIDER_REGISTRY.get(name)
        if not cls:
            logger.warning(f"Unknown STT provider '{name}', defaulting to Groq")
            cls = GroqChunkedSTT
        _provider_cache[name] = cls()

    return _provider_cache[name]
