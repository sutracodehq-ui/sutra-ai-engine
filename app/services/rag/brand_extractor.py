"""
Brand Extractor — analyzes raw text to derive brand identity.

Software Factory: Uses run_pipeline("brand_analyze") for config-driven
LLM execution. ALL behavior — prompts, schemas, validation, driver chains,
timeouts, tenant learning — is defined in intelligence_config.yaml.

This file should NEVER need editing. To change the analysis:
  → Edit intelligence_config.yaml → intelligence_pipelines → brand_analyze
"""

import logging

from app.services.rag.web_crawler import WebCrawler

logger = logging.getLogger(__name__)


class BrandExtractor:
    """Service to extract comprehensive brand identity from a URL.

    Schema is defined entirely in YAML. This code is schema-agnostic —
    it doesn't know or care what fields the analysis returns.
    Tenant learning is handled by run_pipeline() via tenant_id.
    """

    @classmethod
    async def analyze_url(cls, url: str, tenant_id: int | None = None) -> dict | None:
        """
        Fetch URL content and perform deep brand audit.

        Pipeline: fetch HTML → clean text → run_pipeline("brand_analyze")
        Returns whatever schema YAML defines. Never validates fields here.
        Tenant learning: stores result in Qdrant if tenant_id provided.
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

        # 3. Run pipeline — YAML config handles everything:
        #    prompts, driver chain, JSON parsing, field validation, fallbacks, tenant learning
        from app.lib.llm_pipeline import run_pipeline

        result = await run_pipeline("brand_analyze", {
            "url": url,
            "content": raw_text,
        }, tenant_id=tenant_id)

        # Pipeline already validates expected_fields from YAML.
        # We only check "is it a non-empty dict?" — zero schema knowledge.
        if result and isinstance(result, dict) and len(result) > 0:
            return result

        logger.warning(f"BrandExtractor: Pipeline returned no valid data for {url}")
        return None
