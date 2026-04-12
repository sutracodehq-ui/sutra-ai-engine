"""
Multilingual Service — Config-driven language support.

Loads supported languages from config/languages.yaml.
Provides language detection hints and response instructions
that get injected into agent system prompts.
"""

import yaml
import logging
from pathlib import Path
from typing import Optional, List, Tuple

logger = logging.getLogger(__name__)

# ─── Indic Script Ranges & Keywords (Software Factory: Config-driven) ──
_indic_ranges: List[Tuple[int, int, str]] | None = None
_script_labels: dict[str, str] | None = None
_hinglish_words: frozenset[str] | None = None

# ─── Singleton Config ────────────────────────────────────────

_config: dict | None = None


def _load_config() -> dict:
    """Load language configuration from YAML."""
    global _config, _indic_ranges, _script_labels, _hinglish_words
    if _config is None:
        config_path = Path(__file__).resolve().parent.parent.parent.parent / "config" / "languages.yaml"
        try:
            with open(config_path, "r") as f:
                _config = yaml.safe_load(f)
                
            # Initialize fast lookup tables from config
            sr = _config.get("smart_router", {})
            _hinglish_words = frozenset(sr.get("hinglish_words", []))
            raw_ranges = sr.get("indic_ranges", {})
            _indic_ranges = [(v[0], v[1], k) for k, v in raw_ranges.items() if isinstance(v, list) and len(v) == 2]
            _script_labels = sr.get("script_labels", {})
            
        except FileNotFoundError:
            logger.warning(f"Language config not found at {config_path}, using defaults")
            _config = {"languages": {"en": {"name": "English", "script": "Latin"}}, "auto_detect": True}
            _hinglish_words = frozenset()
            _indic_ranges = []
            _script_labels = {}
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


def detect_language(text: str) -> str:
    """
    O(1) language detection with indicator script identification.
    Uses Unicode code point ranges (no heavy ML models).
    """
    if not text:
        return "english"
    
    config = _load_config()
    sr = config.get("smart_router", {})
    sample = text[:sr.get("sample_chars", 100)]
    
    alpha_count = indic_count = 0
    script_counts: dict[str, int] = {}
    
    for char in sample:
        if not char.isalpha():
            continue
        alpha_count += 1
        
        # Fast indicator detect
        cp = ord(char)
        if cp >= 0x0900:
            for lo, hi, script in _indic_ranges:
                if lo <= cp <= hi:
                    indic_count += 1
                    script_counts[script] = script_counts.get(script, 0) + 1
                    break
    
    if alpha_count > 0 and (indic_count / alpha_count) > sr.get("indic_threshold", 0.3):
        if script_counts:
            dominant = max(script_counts, key=script_counts.get)
            return _script_labels.get(dominant, "indic")
        return "indic"
        
    words = text.lower().split()[:sr.get("sample_words", 10)]
    if sum(1 for w in words if w in _hinglish_words) >= sr.get("hinglish_min_matches", 2):
        return "hinglish"
        
    return "english"


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
