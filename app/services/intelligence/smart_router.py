"""
Smart Router — O(1) Decision Tree for driver + model selection.

Software Factory Principle: Config-driven, zero hardcoded data.

All data lives in intelligence_config.yaml → smart_router section:
- Language detection (Indic script names, Hinglish words)
- Complexity signals (complex/simple keyword sets)
- Decision table (27 entries: 3 lengths × 3 signals × 3 agent tiers)
- Driver chains per complexity+language
- Model tiers per driver+complexity

Every operation is O(1):
1. detect_language()  → sample fixed chars + frozenset lookup
2. assess_complexity() → first 3 words + single dict lookup
3. _pick_driver()     → precomputed cache at init
"""

import logging
import unicodedata
from pathlib import Path
from typing import Optional

import yaml

from app.config import get_settings

logger = logging.getLogger(__name__)


# ─── Config Loader (cached singleton) ─────────────────────────

_config_cache: dict | None = None


def _load_config() -> dict:
    """Load smart_router config from YAML. Cached after first call."""
    global _config_cache
    if _config_cache is not None:
        return _config_cache

    config_path = Path("intelligence_config.yaml")
    if not config_path.exists():
        _config_cache = {}
        return _config_cache

    with open(config_path) as f:
        full = yaml.safe_load(f) or {}

    _config_cache = full.get("smart_router", {})
    return _config_cache


def _get(key: str, default=None):
    """O(1) config value read."""
    return _load_config().get(key, default)


# ─── Precomputed Lookup Sets (built once from YAML) ────────────

_indic_scripts: frozenset | None = None
_hinglish_words: frozenset | None = None
_complex_signals: frozenset | None = None
_simple_signals: frozenset | None = None


def _ensure_sets():
    """Build frozensets from YAML lists on first use. O(1) after init."""
    global _indic_scripts, _hinglish_words, _complex_signals, _simple_signals

    if _indic_scripts is not None:
        return  # already built

    cfg = _load_config()
    _indic_scripts = frozenset(cfg.get("indic_scripts", []))
    _hinglish_words = frozenset(cfg.get("hinglish_words", []))
    _complex_signals = frozenset(cfg.get("complex_signals", []))
    _simple_signals = frozenset(cfg.get("simple_signals", []))


# ─── O(1) Language Detection ──────────────────────────────────

def detect_language(text: str) -> str:
    """
    Detect language in O(1) — fixed-size sample, frozenset lookups.

    Returns: 'indic', 'hinglish', or 'english'
    """
    if not text:
        return "english"

    _ensure_sets()
    cfg = _load_config()
    sample_chars = cfg.get("sample_chars", 100)
    sample_words = cfg.get("sample_words", 10)
    indic_threshold = cfg.get("indic_threshold", 0.3)
    hinglish_min = cfg.get("hinglish_min_matches", 2)

    # 1. Sample fixed number of chars — O(1)
    sample = text[:sample_chars]
    alpha_count = 0
    indic_count = 0

    for char in sample:
        if not char.isalpha():
            continue
        alpha_count += 1
        # O(1): unicodedata.name() + startswith against frozenset-backed tuple
        try:
            name = unicodedata.name(char, "")
            # Check if the Unicode name starts with any Indic script prefix
            for script in _indic_scripts:
                if name.startswith(script):
                    indic_count += 1
                    break
        except ValueError:
            pass

    if alpha_count > 0 and (indic_count / alpha_count) > indic_threshold:
        return "indic"

    # 2. Hinglish: check first N words — O(1) with frozenset
    words = text.lower().split()[:sample_words]
    matches = sum(1 for w in words if w in _hinglish_words)
    if matches >= hinglish_min:
        return "hinglish"

    return "english"


# ─── O(1) Complexity Assessment ───────────────────────────────

def _detect_signal(prompt: str) -> str:
    """
    Check first N words for complexity signal. O(1) — frozenset lookup.

    Returns: 'complex', 'simple', or 'none'
    """
    _ensure_sets()
    n = _get("signal_words", 3)

    words = prompt.lower().split(None, n + 1)[:n]

    for w in words:
        if w in _complex_signals:
            return "complex"
    for w in words:
        if w in _simple_signals:
            return "simple"

    return "none"


def _length_bucket(word_count: int) -> str:
    """Classify prompt length. O(1) — two comparisons."""
    buckets = _get("length_buckets", {"short": 10, "long": 80})
    if word_count < buckets.get("short", 10):
        return "short"
    if word_count > buckets.get("long", 80):
        return "long"
    return "medium"


def _get_agent_complexity(agent_type: str) -> str:
    """Read complexity from the agent's YAML config. Cached by hub."""
    try:
        from app.services.agents.hub import get_agent_hub
        hub = get_agent_hub()
        agent = hub.get(agent_type)
        return agent._config.get("complexity", "moderate")
    except Exception:
        return "moderate"


# ─── O(1) Driver Key Check ────────────────────────────────────

def _driver_has_key(driver: str) -> bool:
    """Check if a driver has its API key configured. O(1)."""
    settings = get_settings()
    key_map = {
        "openai": settings.openai_api_key,
        "anthropic": settings.anthropic_api_key,
        "gemini": settings.gemini_api_key,
        "groq": settings.groq_api_key,
        "sarvam": settings.sarvam_api_key,
        "nvidia": settings.nvidia_api_key,
        "ollama": "always_available",
    }
    return bool(key_map.get(driver, ""))


# ─── SmartRouter ───────────────────────────────────────────────

class SmartRouter:
    """
    O(1) Decision Tree Router.

    All data from intelligence_config.yaml. Zero hardcoded constants.

    Flow:
    1. detect_language()     → O(1) sample + frozenset
    2. assess_complexity()   → O(1) first 3 words + dict[key] lookup
    3. _pick_driver()        → O(1) precomputed chain + first available
    4. select_model()        → O(1) dict[driver][complexity] lookup
    """

    def __init__(self, *, enabled: bool = True):
        self._enabled = enabled

    def assess_complexity(self, prompt: str, agent_type: str) -> str:
        """
        O(1) complexity assessment via decision table lookup.

        Features (all O(1)):
        - length_bucket: short | medium | long
        - signal: simple | none | complex
        - agent_tier: from YAML cache
        """
        if not self._enabled:
            return "moderate"

        word_count = len(prompt.split(None, 100))  # cap at 100 splits
        bucket = _length_bucket(word_count)
        signal = _detect_signal(prompt)
        agent_tier = _get_agent_complexity(agent_type)

        # O(1): single dict lookup
        key = f"{bucket}_{signal}_{agent_tier}"
        table = _get("decision_table", {})
        return table.get(key, _get("default_complexity", "moderate"))

    def _pick_driver(self, complexity: str, language: str, circuit_breaker=None) -> str:
        """Pick the first available driver from YAML chain. O(d) worst-case, O(1) typical."""
        lang_key = "indic" if language in ("indic", "hinglish") else "english"

        chains = _get("driver_chains", {})
        chain = chains.get(lang_key, {}).get(complexity, ["ollama", "groq"])

        for driver in chain:
            if not _driver_has_key(driver):
                continue
            if circuit_breaker and not circuit_breaker.is_available(driver):
                logger.debug(f"SmartRouter: {driver} circuit OPEN, skipping")
                continue
            return driver

        return get_settings().ai_driver

    def _pick_model(self, driver: str, complexity: str) -> Optional[str]:
        """O(1): dict[driver][complexity] lookup from YAML."""
        tiers = _get("model_tiers", {})
        driver_tiers = tiers.get(driver, {})
        model = driver_tiers.get(complexity)

        # null in YAML means use env var default
        if model is None:
            settings = get_settings()
            defaults = {
                "ollama": settings.ollama_model,
                "sarvam": settings.sarvam_model,
                "nvidia": settings.nvidia_model,
            }
            model = defaults.get(driver)

        return model

    def select_model(self, prompt: str, agent_type: str, driver: str) -> Optional[str]:
        """Select the optimal model for a prompt within a specific driver."""
        if not self._enabled:
            return None
        complexity = self.assess_complexity(prompt, agent_type)
        return self._pick_model(driver, complexity)

    def route(self, prompt: str, agent_type: str, circuit_breaker=None) -> dict:
        """
        Full O(1) routing decision.

        Returns:
            {
                "driver": "groq",
                "model": "llama-3.3-70b-versatile",
                "complexity": "moderate",
                "language": "english",
                "reason": "moderate task → groq (llama-3.3-70b-versatile)"
            }
        """
        if not self._enabled:
            settings = get_settings()
            return {
                "driver": settings.ai_driver,
                "model": None,
                "complexity": "moderate",
                "language": "english",
                "reason": "smart router disabled",
            }

        # 1. O(1) language detection
        language = detect_language(prompt)

        # 2. O(1) complexity assessment
        complexity = self.assess_complexity(prompt, agent_type)

        # 3. Pick best available driver
        driver = self._pick_driver(complexity, language, circuit_breaker)

        # 4. O(1) model selection
        model = self._pick_model(driver, complexity)

        reason_parts = []
        if language != "english":
            reason_parts.append(f"{language} detected")
        reason_parts.append(f"{complexity} task → {driver}")
        if model:
            reason_parts.append(f"({model})")
        reason = " ".join(reason_parts)

        logger.info(f"SmartRouter: {reason}")

        return {
            "driver": driver,
            "model": model,
            "complexity": complexity,
            "language": language,
            "reason": reason,
        }
