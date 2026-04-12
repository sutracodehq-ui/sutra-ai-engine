"""
Web Crawler — simple, fast extraction of website content.

Software Factory: Crawling is the first step of the RAG pipeline.
This service fetches HTML and performs basic cleaning before analysis.
"""

import logging
import httpx
from typing import Optional

logger = logging.getLogger(__name__)


class WebCrawler:
    """Service to fetch and clean website content."""

    @classmethod
    async def fetch(cls, url: str) -> Optional[str]:
        """Fetch the HTML content of a URL."""
        if not url.startswith("http"):
            url = f"https://{url}"

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/91.0.4472.124 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        }

        try:
            async with httpx.AsyncClient(headers=headers, follow_redirects=True, timeout=10.0) as client:
                response = await client.get(url)
                response.raise_for_status()
                return response.text
        except Exception as e:
            logger.error(f"WebCrawler: failed to fetch {url}: {e}")
            return None

    @classmethod
    def clean_html(cls, html: str) -> str:
        """Strip tags and extra whitespace, keeping essential text block."""
        import re
        
        # Remove scripts and styles
        html = re.sub(r'<(script|style).*?>.*?</\1>', '', html, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove all other tags
        text = re.sub(r'<.*?>', ' ', html)
        
        # Consolidate whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text[:10000]  # Limit to 10k chars for LLM context
