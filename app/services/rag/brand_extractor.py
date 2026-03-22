"""
Brand Extractor — analyzes raw text to derive brand identity.

Uses RAG context (crawled text) + LLM to extract the 'Sutra' (essence)
of a brand from their website.
"""

import logging
from typing import TypedDict, List

from app.services.llm_service import get_llm_service
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

    @classmethod
    async def analyze_url(cls, url: str) -> BrandProfile | None:
        """Fetch URL content and extract brand profile."""
        
        # 1. Fetch
        html = await WebCrawler.fetch(url)
        if not html:
            return None
        
        # 2. Clean
        raw_text = WebCrawler.clean_html(html)
        
        # 3. Analyze with LLM — use default driver chain (no model override)
        service = get_llm_service()

        prompt = f"""
Analyze the following text from the website of {url}.
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
}}
""".strip()

        try:
            result = await service.complete(
                prompt=prompt,
                system_prompt="You are a Brand Strategist. Extract key identity markers from web content.",
                temperature=0.0,
                json_mode=True
            )
            
            import json
            data = json.loads(result.content or "{}")
            return data
        except Exception as e:
            logger.error(f"Brand Extraction failed for {url}: {e}")
            return None
