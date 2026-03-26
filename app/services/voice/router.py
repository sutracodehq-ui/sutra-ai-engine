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
        # 1. Language Detection (Devnagari check)
        has_hindi = any('\u0900' <= char <= '\u097F' for char in text)
        
        # 2. Base Configuration
        edge_config = self.config.get("edge", {})
        voice_map = edge_config.get("voice_map", {})
        default_voice_nickname = requested_voice or self.config.get("default_voice", "nova")
        
        # 3. Provider Selection (Currently defaulting to Edge, with fallback in service)
        # In a more advanced version, this would check circuit breakers
        provider = self.config.get("default_provider", "edge")

        # 4. Voice ID Mapping
        final_voice_id = None
        
        if has_hindi:
            # Smart Gender Switching for Hindi
            if default_voice_nickname in ["nova", "shimmer", "fable"]:
                final_voice_id = voice_map.get("swara", "hi-IN-SwaraNeural")
            else:
                final_voice_id = voice_map.get("madhur", "hi-IN-MadhurNeural")
        else:
            # Map nickname to specific neural model
            final_voice_id = voice_map.get(default_voice_nickname, edge_config.get("default_edge_voice"))

        # 5. Metadata for logging/frontend
        metadata = {
            "requested_nickname": default_voice_nickname,
            "has_hindi": has_hindi,
            "provider": provider,
            "voice_id": final_voice_id,
        }

        logger.info(f"SmartVoiceRouter: {metadata}")
        
        return {
            "provider": provider,
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
