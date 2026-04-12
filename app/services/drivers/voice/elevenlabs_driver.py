"""
ElevenLabs TTS Driver — premium text-to-speech via ElevenLabs API.

Config-driven: reads model, voice_id, and output_format from config/voice.yaml.
Falls back gracefully if ELEVENLABS_API_KEY is not set.
"""

import logging
from typing import Optional

from app.config import get_settings

logger = logging.getLogger(__name__)


class ElevenLabsDriver:
    """
    ElevenLabs text-to-speech driver.

    All config (model, voice_id, output_format) comes from voice.yaml.
    The voice service uses this as the premium TTS provider in the driver chain.
    """

    def __init__(self):
        self._client = None

    def _get_client(self):
        """Lazy-init the ElevenLabs client."""
        if self._client is None:
            settings = get_settings()
            api_key = settings.elevenlabs_api_key
            if not api_key:
                raise RuntimeError("ELEVENLABS_API_KEY not configured")

            from elevenlabs.client import ElevenLabs
            self._client = ElevenLabs(api_key=api_key)

        return self._client

    async def generate_speech(
        self,
        text: str,
        voice_id: str | None = None,
        model_id: str | None = None,
        output_format: str | None = None,
    ) -> bytes:
        """
        Convert text to speech using ElevenLabs API.

        Args:
            text: Text to convert to speech
            voice_id: ElevenLabs voice ID (from YAML config)
            model_id: Model to use (eleven_v3, eleven_flash_v2_5)
            output_format: Output format (mp3_44100_128, etc.)

        Returns:
            Audio bytes (MP3)
        """
        import asyncio
        from functools import partial

        # Read defaults from voice config
        from app.services.voice.config import get_voice_config
        voice_config = get_voice_config()
        el_config = voice_config.get("tts", {}).get("providers", {}).get("elevenlabs", {})

        voice_id = voice_id or el_config.get("default_voice_id", "JBFqnCBsd6RMkjVDRZzb")
        model_id = model_id or el_config.get("model", "eleven_v3")
        output_format = output_format or el_config.get("output_format", "mp3_44100_128")

        client = self._get_client()

        # ElevenLabs SDK is synchronous — wrap in executor
        loop = asyncio.get_event_loop()
        audio_generator = await loop.run_in_executor(
            None,
            partial(
                client.text_to_speech.convert,
                text=text,
                voice_id=voice_id,
                model_id=model_id,
                output_format=output_format,
            ),
        )

        # Collect bytes from generator
        audio_bytes = b""
        for chunk in audio_generator:
            if chunk:
                audio_bytes += chunk

        logger.info(
            f"🔊 ElevenLabs TTS: {len(text)} chars → {len(audio_bytes)} bytes "
            f"(voice={voice_id[:8]}..., model={model_id})"
        )

        return audio_bytes

    def is_available(self) -> bool:
        """Check if ElevenLabs is configured and available."""
        try:
            settings = get_settings()
            return bool(settings.elevenlabs_api_key)
        except Exception:
            return False


# ─── Module-level singleton ──────────────────────────────────

_driver_instance: ElevenLabsDriver | None = None


def get_elevenlabs_driver() -> ElevenLabsDriver:
    """Get or create the ElevenLabs driver singleton."""
    global _driver_instance
    if _driver_instance is None:
        _driver_instance = ElevenLabsDriver()
    return _driver_instance
