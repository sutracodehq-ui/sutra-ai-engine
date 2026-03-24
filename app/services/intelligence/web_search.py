"""
Web Search Service — Real-time internet search for AI context enrichment.

Uses Tavily API (free tier: 1000 searches/month) to give the AI
access to current, real-world information before responding.

How it works:
1. User asks: "What's the latest CBSE syllabus for Physics?"
2. WebSearch.search() → Tavily API → top 5 results with snippets
3. Results injected into LLM context → AI answers with real data

Fallback: if Tavily is unavailable, uses DuckDuckGo HTML scraping (no API key).
"""

import logging
import threading
from typing import Optional

import httpx

from app.config import get_settings

logger = logging.getLogger(__name__)


class WebSearch:
    """
    Real-time web search for AI context enrichment.

    Primary: Tavily API (structured search results, free 1000/mo)
    Fallback: DuckDuckGo HTML (no API key needed, less structured)
    """

    def __init__(self):
        self._client = httpx.AsyncClient(timeout=10, follow_redirects=True)

    async def search(self, query: str, max_results: int = 5) -> dict:
        """
        Search the web and return structured results.

        Returns:
            {
                "results": [{"title": "...", "snippet": "...", "url": "..."}],
                "answer": "AI-generated summary from Tavily (if available)",
                "source": "tavily" | "duckduckgo" | "none"
            }
        """
        settings = get_settings()

        # Try Tavily first (best quality)
        if settings.tavily_api_key:
            result = await self._tavily_search(query, max_results, settings.tavily_api_key)
            if result["results"]:
                return result

        # Fallback: DuckDuckGo instant answer API (no key needed)
        result = await self._ddg_search(query, max_results)
        if result["results"]:
            return result

        return {"results": [], "answer": "", "source": "none"}

    async def _tavily_search(self, query: str, max_results: int, api_key: str) -> dict:
        """Search via Tavily API — structured results + AI summary."""
        try:
            resp = await self._client.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": api_key,
                    "query": query,
                    "max_results": max_results,
                    "include_answer": True,
                    "search_depth": "basic",
                },
            )

            if resp.status_code != 200:
                logger.warning(f"Tavily search failed: HTTP {resp.status_code}")
                return {"results": [], "answer": "", "source": "tavily"}

            data = resp.json()

            results = []
            for item in data.get("results", [])[:max_results]:
                results.append({
                    "title": item.get("title", ""),
                    "snippet": item.get("content", "")[:500],
                    "url": item.get("url", ""),
                })

            return {
                "results": results,
                "answer": data.get("answer", ""),
                "source": "tavily",
            }

        except Exception as e:
            logger.warning(f"Tavily search error: {e}")
            return {"results": [], "answer": "", "source": "tavily"}

    async def _ddg_search(self, query: str, max_results: int) -> dict:
        """Fallback search via DuckDuckGo Instant Answer API (free, no key)."""
        try:
            resp = await self._client.get(
                "https://api.duckduckgo.com/",
                params={"q": query, "format": "json", "no_html": 1, "skip_disambig": 1},
            )

            if resp.status_code != 200:
                return {"results": [], "answer": "", "source": "duckduckgo"}

            data = resp.json()
            results = []

            # Abstract (main answer)
            if data.get("AbstractText"):
                results.append({
                    "title": data.get("Heading", ""),
                    "snippet": data["AbstractText"][:500],
                    "url": data.get("AbstractURL", ""),
                })

            # Related topics
            for topic in data.get("RelatedTopics", [])[:max_results - len(results)]:
                if isinstance(topic, dict) and topic.get("Text"):
                    results.append({
                        "title": topic.get("Text", "")[:100],
                        "snippet": topic.get("Text", "")[:500],
                        "url": topic.get("FirstURL", ""),
                    })

            return {
                "results": results,
                "answer": data.get("AbstractText", ""),
                "source": "duckduckgo",
            }

        except Exception as e:
            logger.debug(f"DuckDuckGo search error: {e}")
            return {"results": [], "answer": "", "source": "duckduckgo"}

    def should_search(self, prompt: str) -> bool:
        """
        Heuristic: Should we search the web for this query?

        Returns True for queries that need current/factual data.
        Returns False for creative/generative tasks and greetings.
        """
        lower = prompt.lower().strip()

        # Too short — greetings, single words
        if len(lower.split()) < 4:
            return False

        # Greetings — never search
        greetings = {"hi", "hello", "hey", "good morning", "good evening", "thanks", "thank you", "bye", "ok", "okay"}
        if lower.rstrip("?!., ") in greetings:
            return False

        # YES — needs web search
        search_signals = {
            "latest", "current", "today", "2024", "2025", "2026",
            "news", "update", "recent", "happening",
            "price", "stock", "market", "weather",
            "who is", "what is", "when is", "where is",
            "how to", "guide", "tutorial", "steps",
            "syllabus", "exam", "result", "notification",
            "government", "policy", "scheme", "regulation",
            "compare", "vs", "difference between",
            "review", "best", "top", "ranking",
        }

        # NO — creative tasks, don't pollute with search
        no_search_signals = {
            "generate", "create", "write", "compose", "draft",
            "quiz", "flashcard", "note", "summary",
            "translate", "paraphrase", "rewrite",
            "code", "function", "class", "api",
        }

        has_search = any(sig in lower for sig in search_signals)
        has_no_search = any(sig in lower for sig in no_search_signals)

        # Search if has search signals and no creative signals
        if has_search and not has_no_search:
            return True

        # Search if it's a question (ends with ?) and long enough
        if prompt.strip().endswith("?") and not has_no_search and len(lower.split()) >= 5:
            return True

        return False

    async def close(self):
        await self._client.aclose()


# ─── Singleton ──────────────────────────────────────────────
_search: WebSearch | None = None
_search_lock = threading.Lock()


def get_web_search() -> WebSearch:
    global _search
    if _search is None:
        with _search_lock:
            if _search is None:
                _search = WebSearch()
    return _search
