import logging

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
        except Exception:
            # Fallback for scripts or outside-of-app calls
            return {}
    return _voice_config
