import logging

logger = logging.getLogger(__name__)

_voice_config = None

def get_voice_config() -> dict:
    """
    Retrieves the voice configuration from intelligence_config.yaml.
    Uses the Brain's configuration loader to ensure single source of truth.
    """
    global _voice_config
    
    if _voice_config is None:
        try:
            from app.services.intelligence.brain import _cfg
            _voice_config = _cfg("voice", default={})
            realtime = _voice_config.get("realtime", {})
            logger.info(
                f"🎙️ Voice config loaded: "
                f"realtime.enabled={realtime.get('enabled', 'MISSING')}, "
                f"stt_provider={realtime.get('stt_provider', 'MISSING')}, "
                f"edge.voice_map keys={list(_voice_config.get('edge', {}).get('voice_map', {}).keys())[:5]}"
            )
        except Exception as e:
            logger.warning(f"Failed to load voice config: {e}")
            return {}
    return _voice_config


def reset_voice_config():
    """Force reload of voice config (after YAML changes)."""
    global _voice_config
    _voice_config = None

