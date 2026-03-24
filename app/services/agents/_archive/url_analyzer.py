"""
UrlAnalyzerAgent — Scrapes a URL and generates a full digital footprint report.

This agent is unique: it has a pre-execution step that runs the WebScraperService
BEFORE sending data to the LLM. The scraped data is injected into the prompt context.

SELF-LEARNING: Every analysis is saved to the `url_analyses` table so the engine
can learn patterns over time (e.g., what tech stacks correlate with high SEO scores).
"""

import json
import logging
from typing import Any
from urllib.parse import urlparse

from app.services.agents.base import BaseAgent
from app.services.intelligence.web_scraper import WebScraperService
from app.services.drivers.base import LlmResponse

logger = logging.getLogger(__name__)


class UrlAnalyzerAgent(BaseAgent):

    @property
    def identifier(self) -> str:
        return "url_analyzer"

    async def execute(self, prompt: str, db: Any = None, context: dict | None = None, **options) -> LlmResponse:
        """
        Override execute to inject scraped data into the LLM prompt.
        
        Flow:
        1. Extract URL from the user prompt.
        2. Run WebScraperService to get raw digital footprint.
        3. Save scraped data to DB for training.
        4. Send scraped data + user prompt to the LLM for analysis.
        5. Save AI report back to DB for fine-tuning.
        """
        # 1. Extract URL from prompt
        url = self._extract_url(prompt)
        
        scraped_data = {}
        if url:
            logger.info(f"UrlAnalyzer: Scraping {url}")
            scraper = WebScraperService()
            scraped_data = await scraper.analyze_url(url, max_pages=5)
        
        # 2. Build enriched prompt with scraped data
        enriched_prompt = f"""
USER REQUEST: {prompt}

--- SCRAPED DIGITAL FOOTPRINT DATA ---
{json.dumps(scraped_data, indent=2, default=str)}
--- END OF SCRAPED DATA ---

Analyze the above scraped data and provide a comprehensive digital footprint report.
Include: SEO health, Google indexing status, tech stack, social presence, 
content quality, structured data, security posture, strengths, weaknesses, 
and prioritized recommendations.
"""
        
        # 3. Generate AI report via the kernel
        response = await super().execute(enriched_prompt, db=db, context=context, **options)
        
        # 4. Save everything to DB for self-learning
        if db and url:
            await self._save_analysis(db, url, scraped_data, response, context)
        
        return response

    async def _save_analysis(self, db, url: str, scraped_data: dict, response: LlmResponse, context: dict | None):
        """Persist the analysis to the database for training."""
        try:
            from app.models.url_analysis import UrlAnalysis
            
            tenant_id = context.get("tenant_id", 0) if context else 0
            seo_score = scraped_data.get("overall_seo_health", {}).get("score", 0)
            security_score = scraped_data.get("security_headers", {}).get("score", 0)
            indexed_pages = scraped_data.get("google_indexing", {}).get("indexed_pages_estimate", 0)
            tech_stack = scraped_data.get("tech_stack", [])
            
            analysis = UrlAnalysis(
                tenant_id=tenant_id,
                url=url,
                domain=urlparse(url).netloc,
                seo_score=seo_score,
                security_score=security_score,
                scraped_data=scraped_data,
                indexed_pages=indexed_pages,
                tech_stack=tech_stack,
                ai_report=response.content if hasattr(response, "content") else str(response),
            )
            
            db.add(analysis)
            await db.commit()
            logger.info(f"UrlAnalyzer: Saved analysis for {url} (SEO={seo_score}, Security={security_score})")
            
        except Exception as e:
            logger.warning(f"UrlAnalyzer: Failed to save analysis: {e}")

    def _extract_url(self, text: str) -> str | None:
        """Extract the first URL from the text."""
        import re
        
        # Match URLs with or without protocol
        pattern = r'https?://[^\s<>"\']+|www\.[^\s<>"\']+|[a-zA-Z0-9][-a-zA-Z0-9]*\.[a-zA-Z]{2,}(?:/[^\s<>"\']*)?'
        match = re.search(pattern, text)
        
        if match:
            url = match.group(0)
            if not url.startswith("http"):
                url = f"https://{url}"
            return url
        return None

