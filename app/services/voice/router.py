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
        Returns: {provider, voice_id, settings}
        """
        # 1. Base Configuration
        edge_config = self.config.get("edge", {})
        voice_map = edge_config.get("voice_map", {})
        requested_nickname = requested_voice or self.config.get("default_voice", "nova")
        
        # 2. Script-based Auto-Detection
        lang_detected = None
        scripts = {
            "tamil": r"[\u0B80-\u0BFF]",
            "telugu": r"[\u0C00-\u0C7F]",
            "kannada": r"[\u0C80-\u0CFF]",
            "malayalam": r"[\u0D00-\u0D7F]",
            "gujarati": r"[\u0A80-\u0AFF]",
            "bengali": r"[\u0980-\u09FF]",
            "marathi": r"[\u0900-\u097F]", # Devnagari (handled below)
            "hindi": r"[\u0900-\u097F]"   # Devnagari
        }

        for lang, pattern in scripts.items():
            if re.search(pattern, text):
                lang_detected = lang
                break

        # 3. Gender Mapping
        gender = "female" # default for nova/shimmer/swara
        if requested_nickname in ["alloy", "echo", "onyx", "madhur"]:
            gender = "male"

        # 4. Final Voice Selection
        final_voice_id = None
        
        if lang_detected:
            if lang_detected == "hindi" or lang_detected == "marathi":
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
            final_voice_id = voice_map.get(requested_nickname, edge_config.get("default_edge_voice"))

        # 5. Metadata for logging/frontend
        metadata = {
            "requested_nickname": requested_nickname,
            "detected_lang": lang_detected,
            "provider": self.config.get("default_provider", "edge"),
            "voice_id": final_voice_id,
        }

        logger.info(f"SmartVoiceRouter: {metadata}")
        
        return {
            "provider": metadata["provider"],
            "voice_id": final_voice_id,
            "metadata": metadata,
            "edge_settings": {
                "rate": edge_config.get("rate", "+0%"),
                "pitch": edge_config.get("pitch", "+0Hz"),
                "volume": edge_config.get("volume", "+0%")
            }
        }

# Singleton instance
_router = None

def get_voice_router() -> SmartVoiceRouter:
    global _router
    if _router is None:
        _router = SmartVoiceRouter()
    return _router
