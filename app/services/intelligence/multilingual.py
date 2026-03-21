"""
Multilingual Service — Config-driven language support.

Loads supported languages from config/languages.yaml.
Provides language detection hints and response instructions
that get injected into agent system prompts.
"""

import yaml
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ─── Singleton Config ────────────────────────────────────────

_config: dict | None = None


def _load_config() -> dict:
    """Load language configuration from YAML."""
    global _config
    if _config is None:
        config_path = Path(__file__).resolve().parent.parent.parent / "config" / "languages.yaml"
        try:
            with open(config_path, "r") as f:
                _config = yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Language config not found at {config_path}, using defaults")
            _config = {"languages": {"en": {"name": "English", "script": "Latin"}}, "auto_detect": True}
    return _config


# ─── Public API ──────────────────────────────────────────────

def get_supported_languages() -> dict:
    """Return all supported languages with metadata."""
    config = _load_config()
    return config.get("languages", {})


def get_language_info(code: str) -> dict | None:
    """Get info for a specific language code."""
    languages = get_supported_languages()
    return languages.get(code)


def get_language_instruction(language_code: Optional[str] = None) -> str:
    """
    Build a language instruction to inject into agent system prompts.

    If language_code is provided → strict instruction in that language.
    If None and auto_detect is True → auto-detect instruction.
    If None and auto_detect is False → empty (defaults to English).
    """
    config = _load_config()

    if language_code and language_code != "en":
        lang = get_language_info(language_code)
        if lang:
            template = config.get("language_instruction", "")
            return template.format(
                language_name=lang["name"],
                language_code=language_code,
                script_name=lang.get("script", "native"),
            )
        else:
            # Unknown code — still try to honor the request
            return f"LANGUAGE RULE: Respond in the language with code '{language_code}'. Use the appropriate script."

    # Auto-detect mode
    if config.get("auto_detect", True):
        return config.get("auto_detect_instruction", "")

    return ""


def list_languages_summary() -> list[dict]:
    """Return a summary list suitable for API responses."""
    languages = get_supported_languages()
    return [
        {
            "code": code,
            "name": info.get("name", code),
            "native_name": info.get("native_name", ""),
            "script": info.get("script", ""),
            "region": info.get("region", ""),
        }
        for code, info in languages.items()
    ]
