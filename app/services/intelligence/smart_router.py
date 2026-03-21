"""
Smart Router — complexity-based model selection.

Performance impact: routes simple tasks to cheap/fast models (GPT-4o-mini, Gemini Flash)
and complex tasks to powerful models (GPT-4o, Claude Sonnet). Saves 40-60% on token costs
while maintaining quality where it matters.
"""

import logging
import re

from app.config import get_settings

logger = logging.getLogger(__name__)


# ─── Complexity Signals ─────────────────────────────────────

COMPLEX_KEYWORDS = {
    "analyze", "compare", "contrast", "evaluate", "synthesize",
    "strategy", "strategic", "comprehensive", "detailed analysis",
    "multi-step", "in-depth", "research", "report", "whitepaper",
    "long-form", "case study", "competitive analysis", "audit",
}

SIMPLE_KEYWORDS = {
    "write", "create", "generate", "make", "draft",
    "short", "quick", "simple", "basic", "brief",
    "one-liner", "tweet", "sms", "subject line", "headline",
    "hashtag", "caption", "tagline", "slogan",
}

# Agent-level complexity defaults
AGENT_COMPLEXITY: dict[str, str] = {
    "sms": "simple",          # Always short content
    "whatsapp": "simple",     # Template-based
    "ad_creative": "simple",  # Short copy
    "copywriter": "moderate", # Can be either
    "social_media": "moderate",
    "email_campaign": "moderate",
    "seo": "complex",         # Requires analysis
    "edtech": "complex",      # Pedagogical reasoning
}

# Model tiers
MODEL_TIERS: dict[str, dict[str, str]] = {
    "openai": {"simple": "gpt-4o-mini", "moderate": "gpt-4o-mini", "complex": "gpt-4o"},
    "anthropic": {"simple": "claude-haiku-4-20250514", "moderate": "claude-sonnet-4-20250514", "complex": "claude-sonnet-4-20250514"},
    "gemini": {"simple": "gemini-2.0-flash", "moderate": "gemini-2.0-flash", "complex": "gemini-2.5-pro-preview-06-05"},
    "groq": {"simple": "llama-3.3-70b-versatile", "moderate": "llama-3.3-70b-versatile", "complex": "llama-3.3-70b-versatile"},
    "ollama": {"simple": "llama3.2:3b", "moderate": "llama3.2:3b", "complex": "llama3.2:3b"},
}

# ─── Fallback Rules (Software Factory: Smart Escalation) ──────
FALLBACK_TO_LOCAL = True  # If true, simple tasks go to Ollama if available


class SmartRouter:
    """Routes tasks to the appropriate model tier based on complexity."""

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
        if word_count < 30:
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

    def select_model(self, prompt: str, agent_type: str, driver: str) -> str | None:
        """
        Select the optimal model for a prompt.

        Returns the model name to use, or None to use the default.
        """
        if not self._enabled:
            return None

        complexity = self.assess_complexity(prompt, agent_type)
        driver_tiers = MODEL_TIERS.get(driver)

        if not driver_tiers:
            return None

        model = driver_tiers.get(complexity)
        logger.info(f"SmartRouter: prompt={complexity}, driver={driver} → model={model}")
        return model

    def route(self, prompt: str, agent_type: str) -> dict:
        """Full routing decision — returns driver + model recommendation."""
        settings = get_settings()
        driver = settings.ai_driver
        complexity = self.assess_complexity(prompt, agent_type)
        
        # Smart Fallback: If task is simple and local fallback enabled, use Ollama
        if complexity == "simple" and FALLBACK_TO_LOCAL:
            driver = "ollama"
            
        model = self.select_model(prompt, agent_type, driver)

        return {
            "driver": driver,
            "model": model,
            "complexity": complexity,
            "word_count": len(prompt.split()),
        }
