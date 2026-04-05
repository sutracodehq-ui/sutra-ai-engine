"""
Real-Time STT — Provider-abstracted speech-to-text for live audio.

Software Factory:
- Config-driven: provider selection from config/voice.yaml → realtime.stt_provider
- Provider abstraction: GroqChunkedSTT (default) + OpenAI Whisper (fallback)
- No hardcoded URLs, models, or thresholds

Pipeline: audio bytes → provider → {"text": "...", "language": "hi", "duration": 2.5}
"""

import io
import logging
from abc import ABC, abstractmethod
from typing import Optional

import httpx

from app.config import get_settings
from app.services.voice.config import get_voice_config

logger = logging.getLogger(__name__)


class RealtimeSTTProvider(ABC):
    """Abstract base for real-time STT providers."""

    @abstractmethod
    async def transcribe(
        self,
        audio_bytes: bytes,
        language_hint: Optional[str] = None,
        content_type: str = "audio/webm",
    ) -> dict:
        """
        Transcribe audio bytes.

        Returns: {"text": "...", "language": "hi", "duration": 2.5}
        """
        ...


class GroqChunkedSTT(RealtimeSTTProvider):
    """
    Groq Whisper — blazing-fast REST-based STT.

    Sends audio chunks to Groq's Whisper API endpoint.
    Uses OpenAI-compatible /v1/audio/transcriptions format.
    Free tier, ~300ms latency, supports 99+ languages.
    """

    def __init__(self):
        voice_cfg = get_voice_config()
        realtime_cfg = voice_cfg.get("realtime", {})
        self._groq_cfg = realtime_cfg.get("groq", {})

    async def transcribe(
        self,
        audio_bytes: bytes,
        language_hint: Optional[str] = None,
        content_type: str = "audio/webm",
    ) -> dict:
        settings = get_settings()
        api_key = settings.groq_api_key
        if not api_key:
            raise ValueError("GROQ_API_KEY not configured for real-time STT")

        model = self._groq_cfg.get("model", "whisper-large-v3-turbo")
        temperature = self._groq_cfg.get("temperature", 0.0)

        # Determine file extension from content type
        ext_map = {
            "audio/webm": "webm",
            "audio/wav": "wav",
            "audio/mp3": "mp3",
            "audio/mpeg": "mp3",
            "audio/ogg": "ogg",
            "audio/flac": "flac",
        }
        ext = ext_map.get(content_type, "webm")
        filename = f"chunk.{ext}"

        form_data = {
            "model": (None, model),
            "response_format": (None, "verbose_json"),
            "temperature": (None, str(temperature)),
        }

        # Language hint (improves accuracy for known language)
        hint = language_hint or self._groq_cfg.get("language_hint")
        if hint:
            form_data["language"] = (None, hint)

        # Initial prompt — biases Whisper toward expected languages/vocabulary
        # This is the key trick for Hindi/Hinglish/multilingual accuracy
        initial_prompt = self._groq_cfg.get("initial_prompt")
        if initial_prompt:
            form_data["prompt"] = (None, initial_prompt)

        files = {
            "file": (filename, audio_bytes, content_type),
        }

        # Build URL from config (never hardcode — Software Factory rule)
        from app.services.intelligence.brain import _cfg
        providers_cfg = _cfg("providers", default={})
        groq_base_url = providers_cfg.get("groq", {}).get("base_url", "https://api.groq.com/openai/v1")
        transcription_url = f"{groq_base_url.rstrip('/')}/audio/transcriptions"

        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                transcription_url,
                headers={"Authorization": f"Bearer {api_key}"},
                data={k: v[1] for k, v in form_data.items()},
                files=files,
            )

            if response.status_code != 200:
                logger.error(f"Groq Whisper error: {response.status_code} - {response.text[:200]}")
                raise ValueError(f"Groq STT failed ({response.status_code}): {response.text[:100]}")

            result = response.json()
            text = result.get("text", "").strip()
            language = result.get("language", "unknown")
            duration = result.get("duration", 0)

            logger.info(
                f"🎤 Groq STT: '{text[:60]}...' "
                f"(lang={language}, dur={duration:.1f}s, model={model})"
            )

            return {
                "text": text,
                "language": language,
                "duration": duration,
            }


class OpenAIBatchSTT(RealtimeSTTProvider):
    """
    OpenAI Whisper — batch fallback STT.

    Used when Groq is unavailable. Same Whisper API as existing voice_service.
    """

    def __init__(self):
        voice_cfg = get_voice_config()
        realtime_cfg = voice_cfg.get("realtime", {})
        self._oai_cfg = realtime_cfg.get("openai_batch", {})

    async def transcribe(
        self,
        audio_bytes: bytes,
        language_hint: Optional[str] = None,
        content_type: str = "audio/webm",
    ) -> dict:
        # Reuse the existing transcribe_audio function
        from app.services.voice.voice_service import transcribe_audio

        ext_map = {"audio/webm": "webm", "audio/wav": "wav", "audio/mp3": "mp3"}
        ext = ext_map.get(content_type, "webm")

        return await transcribe_audio(audio_bytes, f"chunk.{ext}", language_hint)


# ─── Factory ─────────────────────────────────────────────────

_provider_cache: dict[str, RealtimeSTTProvider] = {}


def get_realtime_stt(provider_override: Optional[str] = None) -> RealtimeSTTProvider:
    """
    Factory — returns the correct STT provider from YAML config.

    Config path: config/voice.yaml → realtime.stt_provider
    """
    voice_cfg = get_voice_config()
    realtime_cfg = voice_cfg.get("realtime", {})
    provider_name = provider_override or realtime_cfg.get("stt_provider", "groq")

    if provider_name not in _provider_cache:
        if provider_name == "groq":
            _provider_cache[provider_name] = GroqChunkedSTT()
        elif provider_name in ("openai", "openai_batch"):
            _provider_cache[provider_name] = OpenAIBatchSTT()
        else:
            logger.warning(f"Unknown STT provider '{provider_name}', defaulting to Groq")
            _provider_cache[provider_name] = GroqChunkedSTT()

    return _provider_cache[provider_name]
