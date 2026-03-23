"""
Smart Router — auto-detect language, complexity, and task type to pick the best driver + model.

This is the brain of the multi-model engine:
1. Language Detection: Hindi/Indic → Sarvam, English → default chain
2. Complexity Scoring: simple → Ollama (local), moderate → Groq (fast cloud), complex → NVIDIA/Claude
3. Agent Awareness: quiz_generator always gets a powerful model, sms gets a lightweight one
4. Circuit Breaker Aware: skips drivers that are currently failing

No LLM calls needed — pure heuristic detection. Runs in <1ms.
"""

import logging
import re
import unicodedata

from app.config import get_settings

logger = logging.getLogger(__name__)


# ─── Language Detection ──────────────────────────────────────

# Unicode ranges for Indian scripts
INDIC_RANGES = [
    (0x0900, 0x097F),  # Devanagari (Hindi, Marathi, Sanskrit)
    (0x0980, 0x09FF),  # Bengali
    (0x0A00, 0x0A7F),  # Gurmukhi (Punjabi)
    (0x0A80, 0x0AFF),  # Gujarati
    (0x0B00, 0x0B7F),  # Odia
    (0x0B80, 0x0BFF),  # Tamil
    (0x0C00, 0x0C7F),  # Telugu
    (0x0C80, 0x0CFF),  # Kannada
    (0x0D00, 0x0D7F),  # Malayalam
]

# Common Hindi words in Latin script (Hinglish detection)
HINGLISH_WORDS = {
    "kya", "hai", "kaise", "karo", "batao", "mujhe", "banao", "chahiye",
    "acha", "theek", "nahi", "haan", "bhai", "yaar", "dost", "kuch",
    "mera", "tera", "humara", "aapka", "kaisa", "kitna", "kab", "kahan",
    "kyun", "kaise", "samjhao", "padhao", "likho", "bolo", "dekho",
    "namaste", "dhanyavaad", "shukriya", "bahut", "accha", "thik",
}


def detect_language(text: str) -> str:
    """
    Detect if text is primarily Indic or English.

    Returns: 'indic', 'hinglish', or 'english'
    """
    if not text:
        return "english"

    # Count Indic script characters
    indic_chars = 0
    total_chars = 0

    for char in text:
        cp = ord(char)
        if char.isalpha():
            total_chars += 1
            for start, end in INDIC_RANGES:
                if start <= cp <= end:
                    indic_chars += 1
                    break

    # If >30% Indic script characters → Indic
    if total_chars > 0 and (indic_chars / total_chars) > 0.3:
        return "indic"

    # Check for Hinglish (Hindi words in Latin script)
    words = set(text.lower().split())
    hinglish_matches = words & HINGLISH_WORDS
    if len(hinglish_matches) >= 2:
        return "hinglish"

    return "english"


# ─── Complexity Signals ──────────────────────────────────────

COMPLEX_KEYWORDS = {
    "analyze", "compare", "contrast", "evaluate", "synthesize",
    "strategy", "strategic", "comprehensive", "detailed analysis",
    "multi-step", "in-depth", "research", "report", "whitepaper",
    "long-form", "case study", "competitive analysis", "audit",
    "explain in detail", "step by step", "elaborate",
}

SIMPLE_KEYWORDS = {
    "short", "quick", "simple", "basic", "brief",
    "one-liner", "tweet", "sms", "subject line", "headline",
    "hashtag", "caption", "tagline", "slogan", "hi", "hello",
    "thanks", "thank you", "ok", "okay", "bye",
}

# Agent-level complexity defaults
AGENT_COMPLEXITY: dict[str, str] = {
    # Simple/fast agents
    "sms": "simple",
    "whatsapp": "simple",
    "chatbot_trainer": "moderate",
    # Moderate agents
    "ad_creative": "moderate",
    "copywriter": "moderate",
    "social_media": "moderate",
    "email_campaign": "moderate",
    "content_repurposer": "moderate",
    "note_generator": "moderate",
    "flashcard_creator": "moderate",
    "student_data_validator": "moderate",
    "document_ocr_extractor": "moderate",
    "medicine_info": "moderate",
    "ayurveda_advisor": "moderate",
    # Complex agents (need powerful models)
    "quiz_generator": "complex",
    "lecture_planner": "complex",
    "seo": "complex",
    "brand_auditor": "complex",
    "competitor_analyst": "complex",
    "trend_spotter": "complex",
    "brand_advisor": "complex",
    "education_guru": "complex",
    "edtech": "complex",
    "udise_compliance_advisor": "complex",
    "infrastructure_auditor": "complex",
    "udise_report_generator": "complex",
    "tax_advisor": "complex",
    "mutual_fund_advisor": "complex",
    "insurance_advisor": "complex",
    "symptom_checker": "complex",
    "contract_analyzer": "complex",
    "stock_analyzer": "complex",
}


# ─── Driver Priority Chain (per complexity tier) ─────────────
# Each tier has an ordered list of drivers to try.
# The router picks the first driver that has a valid API key configured.

def _get_driver_chain() -> dict[str, list[str]]:
    """
    Complexity → ordered driver preferences.
    Local first (ollama) → cloud fallback.
    First available (has API key) wins.
    """
    return {
        "simple": ["ollama", "groq", "gemini"],
        "moderate": ["ollama", "groq", "gemini", "openai"],
        "complex": ["ollama", "nvidia", "anthropic", "openai", "gemini", "groq"],
    }


def _get_indic_driver_chain() -> dict[str, list[str]]:
    """Driver chain when language is Indic — Sarvam gets priority after local."""
    return {
        "simple": ["ollama", "sarvam", "groq", "gemini"],
        "moderate": ["ollama", "sarvam", "groq", "gemini", "openai"],
        "complex": ["ollama", "sarvam", "nvidia", "anthropic", "openai", "gemini"],
    }


# Model tiers per driver
def _get_model_tiers() -> dict[str, dict[str, str]]:
    """Build model tiers using config values."""
    settings = get_settings()
    return {
        "openai": {"simple": "gpt-4o-mini", "moderate": "gpt-4o-mini", "complex": "gpt-4o"},
        "anthropic": {"simple": "claude-haiku-4-20250514", "moderate": "claude-sonnet-4-20250514", "complex": "claude-sonnet-4-20250514"},
        "gemini": {"simple": "gemini-2.0-flash", "moderate": "gemini-2.0-flash", "complex": "gemini-2.5-pro-preview-06-05"},
        "groq": {"simple": "llama-3.3-70b-versatile", "moderate": "llama-3.3-70b-versatile", "complex": "llama-3.3-70b-versatile"},
        "ollama": {"simple": settings.ollama_model, "moderate": settings.ollama_model, "complex": settings.ollama_model},
        "sarvam": {"simple": settings.sarvam_model, "moderate": settings.sarvam_model, "complex": settings.sarvam_model},
        "nvidia": {"simple": "meta/llama-3.1-8b-instruct", "moderate": "meta/llama-3.1-70b-instruct", "complex": settings.nvidia_model},
    }


# ─── API Key Availability Check ──────────────────────────────

def _driver_has_key(driver: str) -> bool:
    """Check if a driver has its API key configured (non-empty)."""
    settings = get_settings()
    key_map = {
        "openai": settings.openai_api_key,
        "anthropic": settings.anthropic_api_key,
        "gemini": settings.gemini_api_key,
        "groq": settings.groq_api_key,
        "sarvam": settings.sarvam_api_key,
        "nvidia": settings.nvidia_api_key,
        "ollama": "always_available",  # Local, no key needed
    }
    return bool(key_map.get(driver, ""))


class SmartRouter:
    """
    Auto-detect language + complexity → pick best driver + model.

    Usage:
        router = SmartRouter()
        decision = router.route("मुझे law of motion पर quiz बनाओ", "chatbot_trainer")
        # → {"driver": "sarvam", "model": "sarvam-m", "complexity": "complex",
        #    "language": "hinglish", "reason": "hinglish detected + complex task"}

        decision = router.route("Generate a quiz on law of motion", "education_guru")
        # → {"driver": "nvidia", "model": "meta/llama-3.1-405b-instruct", "complexity": "complex",
        #    "language": "english", "reason": "complex agent + powerful model needed"}
    """

    def __init__(self, *, enabled: bool = True):
        self._enabled = enabled

    def assess_complexity(self, prompt: str, agent_type: str) -> str:
        """
        Score prompt complexity: simple | moderate | complex

        Factors:
        1. Prompt length
        2. Keyword signals
        3. Agent type default
        4. Structural complexity (lists, multi-part instructions)
        """
        if not self._enabled:
            return "moderate"

        score = 0

        # 1. Prompt length
        word_count = len(prompt.split())
        if word_count < 20:
            score -= 1
        elif word_count > 150:
            score += 2
        elif word_count > 80:
            score += 1

        # 2. Keyword signals
        lower = prompt.lower()
        for kw in COMPLEX_KEYWORDS:
            if kw in lower:
                score += 1
        for kw in SIMPLE_KEYWORDS:
            if kw in lower:
                score -= 1

        # 3. Agent type default
        agent_default = AGENT_COMPLEXITY.get(agent_type, "moderate")
        if agent_default == "complex":
            score += 1
        elif agent_default == "simple":
            score -= 1

        # 4. Structural complexity (numbered lists, multiple questions)
        if re.search(r"\d+\.\s", prompt):
            score += 1  # Has numbered lists
        if prompt.count("?") > 2:
            score += 1  # Multiple questions

        # Map score to tier
        if score <= -1:
            return "simple"
        elif score >= 2:
            return "complex"
        return "moderate"

    def _pick_driver(self, complexity: str, language: str, circuit_breaker=None) -> str:
        """Pick the first available driver from the priority chain."""
        if language in ("indic", "hinglish"):
            chains = _get_indic_driver_chain()
        else:
            chains = _get_driver_chain()

        chain = chains.get(complexity, chains["moderate"])

        for driver in chain:
            # Skip if no API key
            if not _driver_has_key(driver):
                continue

            # Skip if circuit breaker says it's down
            if circuit_breaker and not circuit_breaker.is_available(driver):
                logger.debug(f"SmartRouter: {driver} circuit OPEN, skipping")
                continue

            return driver

        # Fallback: use whatever is configured as ai_driver
        return get_settings().ai_driver

    def select_model(self, prompt: str, agent_type: str, driver: str) -> str | None:
        """Select the optimal model for a prompt within a specific driver."""
        if not self._enabled:
            return None

        complexity = self.assess_complexity(prompt, agent_type)
        driver_tiers = _get_model_tiers().get(driver)

        if not driver_tiers:
            return None

        model = driver_tiers.get(complexity)
        return model

    def route(self, prompt: str, agent_type: str, circuit_breaker=None) -> dict:
        """
        Full auto-routing decision — detects language + complexity → picks driver + model.

        Returns:
            {
                "driver": "nvidia",
                "model": "meta/llama-3.1-405b-instruct",
                "complexity": "complex",
                "language": "english",
                "reason": "complex agent, powerful model needed"
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

        # 1. Detect language
        language = detect_language(prompt)

        # 2. Assess complexity
        complexity = self.assess_complexity(prompt, agent_type)

        # 3. Pick best available driver
        driver = self._pick_driver(complexity, language, circuit_breaker)

        # 4. Pick model tier within that driver
        model_tiers = _get_model_tiers().get(driver, {})
        model = model_tiers.get(complexity)

        reason_parts = []
        if language != "english":
            reason_parts.append(f"{language} detected")
        reason_parts.append(f"{complexity} task")
        reason_parts.append(f"→ {driver}")
        if model:
            reason_parts.append(f"({model})")
        reason = ", ".join(reason_parts)

        logger.info(f"SmartRouter: {reason}")

        return {
            "driver": driver,
            "model": model,
            "complexity": complexity,
            "language": language,
            "reason": reason,
        }
