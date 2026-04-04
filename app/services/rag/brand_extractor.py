"""
Brand Extractor — analyzes raw text to derive brand identity.

Uses the Hybrid Router for intelligent model selection:
  1. Try local Ollama first (fast, free)
  2. Quality gate → if bad, escalate to cloud (Groq → Gemini)
  3. Auto-trains local from cloud successes

For brand analysis, we use cloud-first since JSON extraction
needs high accuracy and users are waiting on the UI.
"""

import json
import logging
from typing import TypedDict, List

from app.services.rag.web_crawler import WebCrawler

logger = logging.getLogger(__name__)


class BrandProfile(TypedDict):
    name: str
    mission: str
    voice_tone: str
    target_audience: str
    core_values: List[str]
    slogan: str


class BrandExtractor:
    """Service to extract brand identity from a URL."""

    # Expected fields for quality gate (helps HybridRouter score responses)
    _EXPECTED_FIELDS = ["name", "mission", "voice_tone", "target_audience", "core_values", "slogan"]

    @classmethod
    async def analyze_url(cls, url: str) -> BrandProfile | None:
        """Fetch URL content and extract brand profile using hybrid routing."""

        # 1. Fetch
        html = await WebCrawler.fetch(url)
        if not html:
            logger.warning(f"BrandExtractor: Failed to fetch {url}")
            return None

        # 2. Clean
        raw_text = WebCrawler.clean_html(html)
        if len(raw_text.strip()) < 50:
            logger.warning(f"BrandExtractor: Insufficient content from {url} ({len(raw_text)} chars)")
            return None

        # 3. Build prompt
        system_prompt = "You are a Brand Strategist. Extract key identity markers from web content. Always respond with valid JSON only."
        prompt = f"""Analyze the following text from the website of {url}.
Your goal is to extract the brand identity of this company.

TEXT SAMPLES:
{raw_text[:4000]}

### Output Format:
Return ONLY a JSON object with this schema:
{{
  "name": "Company Name",
  "mission": "One sentence mission statement",
  "voice_tone": "Description of writing style (e.g., 'professional but quirky', 'ultra-minimal')",
  "target_audience": "Who they are selling to",
  "core_values": ["Value 1", "Value 2", "Value 3"],
  "slogan": "Primary tagline/slogan"
}}"""

        # 4. Use cloud-first for brand analysis (accuracy matters, user is waiting)
        #    Falls back through: Groq → Gemini → Ollama
        try:
            data = await cls._extract_with_cloud_first(prompt, system_prompt)
            if data:
                return data
        except Exception as e:
            logger.warning(f"BrandExtractor: Cloud-first failed: {e}")

        # 5. Last resort: try hybrid router (local → cloud escalation)
        try:
            data = await cls._extract_with_hybrid(prompt, system_prompt)
            if data:
                return data
        except Exception as e:
            logger.error(f"BrandExtractor: Hybrid extraction failed for {url}: {e}")

        return None

    @classmethod
    async def _extract_with_cloud_first(cls, prompt: str, system_prompt: str) -> dict | None:
        """
        Fast cloud-first extraction: Groq (free) → Gemini → Anthropic.
        Best for user-facing endpoints where speed + accuracy matter.
        """
        from app.services.intelligence.driver import get_driver_registry

        registry = get_driver_registry()

        # Ordered by speed and cost: Groq (free, fast) → Gemini → Anthropic
        cloud_drivers = ["groq", "gemini", "anthropic"]

        for driver_name in cloud_drivers:
            if not registry.circuit_breaker.is_available(driver_name):
                logger.debug(f"BrandExtractor: {driver_name} circuit OPEN, skipping")
                continue

            try:
                response = await registry.complete(
                    system_prompt=system_prompt,
                    user_prompt=prompt,
                    driver_override=driver_name,
                    temperature=0.0,
                    json_mode=True,
                )

                if response.content:
                    data = json.loads(response.content)
                    if isinstance(data, dict) and "name" in data:
                        logger.info(
                            f"BrandExtractor: SUCCESS via {driver_name} "
                            f"({response.total_tokens} tokens)"
                        )
                        return data

            except json.JSONDecodeError:
                logger.warning(f"BrandExtractor: {driver_name} returned invalid JSON")
                continue
            except Exception as e:
                logger.warning(f"BrandExtractor: {driver_name} failed: {e}")
                continue

        return None

    @classmethod
    async def _extract_with_hybrid(cls, prompt: str, system_prompt: str) -> dict | None:
        """Fallback: use HybridRouter (local → cloud escalation)."""
        from app.services.intelligence.hybrid_router import get_hybrid_router

        router = get_hybrid_router()
        response = await router.execute(
            prompt=prompt,
            system_prompt=system_prompt,
            agent_type="brand_extractor",
            expected_fields=cls._EXPECTED_FIELDS,
            temperature=0.0,
            json_mode=True,
        )

        if response.content:
            try:
                data = json.loads(response.content)
                if isinstance(data, dict) and "name" in data:
                    return data
            except json.JSONDecodeError:
                logger.warning(f"BrandExtractor: hybrid returned invalid JSON")

        return None
