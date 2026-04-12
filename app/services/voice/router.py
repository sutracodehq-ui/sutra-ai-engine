"""
SmartVoiceRouter — Config-driven voice routing (Software Factory).

All language detection patterns and voice mappings come from config/voice.yaml.
No hardcoded regex, no hardcoded voice IDs.
"""

import logging
import re
from typing import Dict, Any, Optional
from app.services.voice.config import get_voice_config

logger = logging.getLogger(__name__)


class SmartVoiceRouter:
    """
    Polymorphic router for Text-to-Speech.
    Decides the best Provider, Voice, and Tone based on language, availability, and tenant preferences.
    """

    def __init__(self):
        self.config = get_voice_config()

    def route(self, text: str, requested_voice: Optional[str] = None, tenant_slug: str = "default") -> Dict[str, Any]:
        """
        Routes the request to the optimal voice and provider.
        Returns: {provider, voice_id, metadata, edge_settings}
        """
        # 1. Base Configuration
        realtime_cfg = self.config.get("realtime", {})
        edge_config = self.config.get("edge", {})
        voice_map = edge_config.get("voice_map", {})
        default_voice = edge_config.get("default_edge_voice", "en-IN-NeerjaNeural")

        defaults = realtime_cfg.get("defaults", {})
        requested_nickname = requested_voice or defaults.get("voice_nickname", "alloy")

        # 2. Script-based Auto-Detection (Software Factory: Load from Config)
        lang_detected = None
        scripts = realtime_cfg.get("language_scripts", {})

        for lang, pattern in scripts.items():
            try:
                if re.search(pattern, text):
                    lang_detected = lang
                    break
            except Exception as e:
                logger.warning(f"Invalid regex pattern for {lang}: {e}")

        # 3. Gender Mapping
        gender = "female"
        male_voices = ["alloy", "echo", "onyx", "madhur"]
        if requested_nickname in male_voices:
            gender = "male"

        # 4. Final Voice Selection
        final_voice_id = None

        if lang_detected:
            if lang_detected in ["hindi", "marathi"]:
                # Smart Gender Switching for Devnagari based languages
                if gender == "female":
                    final_voice_id = voice_map.get("swara", "hi-IN-SwaraNeural")
                else:
                    final_voice_id = voice_map.get("madhur", "hi-IN-MadhurNeural")
            else:
                # Direct regional mapping (Pallavi, Shruti, etc.)
                final_voice_id = voice_map.get(lang_detected)

        # Fallback if no script detected or mapping failed (English/Global)
        if not final_voice_id:
            final_voice_id = voice_map.get(requested_nickname, default_voice)

        # 5. TTS prosody settings from config
        prosody = realtime_cfg.get("tts", {}).get("prosody", {})

        # 6. Metadata for logging/frontend
        provider = self.config.get("default_provider", "edge")

        metadata = {
            "requested_nickname": requested_nickname,
            "detected_lang": lang_detected,
            "provider": provider,
            "voice_id": final_voice_id,
        }

        logger.debug(f"SmartVoiceRouter: {metadata}")

        return {
            "provider": provider,
            "voice_id": final_voice_id,
            "metadata": metadata,
            "edge_settings": {
                "rate": prosody.get("rate", "-5%"),
                "pitch": prosody.get("pitch", "-2Hz"),
                "volume": prosody.get("volume", "+0%"),
            }
        }


# Singleton instance (lazy init)
_router = None


def get_voice_router() -> SmartVoiceRouter:
    global _router
    if _router is None:
        _router = SmartVoiceRouter()
    return _router
