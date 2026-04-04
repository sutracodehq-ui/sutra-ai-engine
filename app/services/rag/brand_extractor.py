"""
Brand Extractor — analyzes raw text to derive brand identity.

Software Factory: Uses run_pipeline("brand_analyze") for config-driven
LLM execution. All prompts, driver chains, timeouts, and expected fields
are defined in intelligence_config.yaml → intelligence_pipelines.
"""

import logging
from typing import TypedDict, List

from app.services.rag.web_crawler import WebCrawler

logger = logging.getLogger(__name__)


class BrandProfile(TypedDict, total=False):
    """Comprehensive brand audit result — schema defined in YAML."""
    brand_identity: dict
    voice_and_tone: dict
    target_audience: dict
    products_and_services: dict
    industry_and_positioning: dict
    digital_presence: dict
    visual_identity: dict
    content_strategy: dict
    social_proof: dict
    technology_signals: dict
    geographic_presence: dict
    core_values: List[str]
    strengths: List[str]
    weaknesses_or_gaps: List[str]
    overall_brand_score: dict


class BrandExtractor:
    """Service to extract comprehensive brand identity from a URL."""

    @classmethod
    async def analyze_url(cls, url: str) -> dict | None:
        """
        Fetch URL content and perform deep brand audit.

        Pipeline: fetch HTML → clean text → run_pipeline("brand_analyze")
        Returns a rich 15-section brand analysis from YAML-defined prompts.
        """
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

        # 3. Run pipeline — config handles prompts, drivers, parsing, validation
        from app.lib.llm_pipeline import run_pipeline

        result = await run_pipeline("brand_analyze", {
            "url": url,
            "content": raw_text,
        })

        if result and isinstance(result, dict) and "brand_identity" in result:
            return result

        logger.warning(f"BrandExtractor: Pipeline returned no valid brand data for {url}")
        return None
