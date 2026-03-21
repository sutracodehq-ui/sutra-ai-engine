"""
Web Intelligence Scanner — Continuous internet learning engine.

Scans RSS feeds, news APIs, and financial data sources to keep
the AI system's knowledge current. Stores insights in ChromaDB
for instant recall by any agent.

Sources:
- RSS: TechCrunch AI, MIT AI News, HackerNews, AI blogs
- Finance: Yahoo Finance API (free), Alpha Vantage
- Trends: Google Trends (via pytrends), Reddit
"""

import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Optional

import httpx

from app.config import get_settings

logger = logging.getLogger(__name__)

# ─── Feed Sources (loaded from scanner_feeds.yaml) ──────────

def _load_feeds() -> dict[str, list[dict]]:
    """Load RSS feed sources from YAML config."""
    import yaml
    from pathlib import Path

    feeds_path = Path("scanner_feeds.yaml")
    if not feeds_path.exists():
        logger.warning("scanner_feeds.yaml not found, using empty feeds")
        return {}

    with open(feeds_path) as f:
        return yaml.safe_load(f) or {}

# ─── Intelligence Config (loaded from intelligence_config.yaml) ─

def _load_intelligence_config() -> dict:
    """Load intelligence module config from YAML."""
    import yaml
    from pathlib import Path

    config_path = Path("intelligence_config.yaml")
    if not config_path.exists():
        logger.warning("intelligence_config.yaml not found, using defaults")
        return {}

    with open(config_path) as f:
        return yaml.safe_load(f) or {}


class WebScanner:
    """
    Continuous internet intelligence scanner.
    
    Fetches data from RSS feeds, stock APIs, and news sources.
    Stores everything in ChromaDB for agent context injection.
    """

    def __init__(self):
        self._client = httpx.AsyncClient(timeout=15, follow_redirects=True)
        self._chroma = None

    def _get_chroma(self):
        """Lazy-init ChromaDB client."""
        if self._chroma is None:
            import chromadb
            settings = get_settings()
            self._chroma = chromadb.HttpClient(host=settings.chromadb_url.replace("http://", "").split(":")[0],
                                                port=int(settings.chromadb_url.split(":")[-1]))
        return self._chroma

    # ─── RSS Scanner ────────────────────────────────────────

    async def scan_rss(self, category: str = "ai_trends") -> list[dict]:
        """Scan RSS feeds for a category and return parsed articles."""
        import xml.etree.ElementTree as ET

        all_feeds = _load_feeds()
        feeds = all_feeds.get(category, [])
        articles = []

        for feed in feeds:
            try:
                resp = await self._client.get(feed["url"], headers={"User-Agent": "SutraAI/1.0"})
                if resp.status_code != 200:
                    continue

                root = ET.fromstring(resp.text)
                # Handle both RSS 2.0 and Atom feeds
                items = root.findall(".//item") or root.findall(".//{http://www.w3.org/2005/Atom}entry")

                for item in items[:10]:  # Latest 10 per feed
                    title = self._get_text(item, "title") or self._get_text(item, "{http://www.w3.org/2005/Atom}title")
                    desc = self._get_text(item, "description") or self._get_text(item, "{http://www.w3.org/2005/Atom}summary")
                    link = self._get_text(item, "link") or self._get_attr(item, "{http://www.w3.org/2005/Atom}link", "href")
                    pub_date = self._get_text(item, "pubDate") or self._get_text(item, "{http://www.w3.org/2005/Atom}published")

                    if title:
                        # Strip HTML tags from description
                        import re
                        clean_desc = re.sub(r'<[^>]+>', '', desc or '')[:500]

                        articles.append({
                            "title": title,
                            "description": clean_desc,
                            "link": link or "",
                            "source": feed["name"],
                            "category": category,
                            "published": pub_date or "",
                            "scanned_at": datetime.now(timezone.utc).isoformat(),
                        })

                logger.info(f"WebScanner: {feed['name']} → {len(items[:10])} articles")

            except Exception as e:
                logger.warning(f"WebScanner: RSS feed '{feed['name']}' failed: {e}")

        return articles

    # ─── Stock Data Scanner ─────────────────────────────────

    async def scan_stocks(self, symbols: list[str] | None = None) -> list[dict]:
        """
        Fetch stock prices from Yahoo Finance API (free, no key needed).
        """
        config = _load_intelligence_config()
        symbols = symbols or config.get("stock_symbols", ["AAPL", "GOOGL", "MSFT"])
        quotes = []

        try:
            # Yahoo Finance v8 API — free, no auth
            symbols_str = ",".join(symbols)
            url = f"https://query1.finance.yahoo.com/v7/finance/quote?symbols={symbols_str}"
            resp = await self._client.get(url, headers={"User-Agent": "SutraAI/1.0"})

            if resp.status_code == 200:
                data = resp.json()
                for quote in data.get("quoteResponse", {}).get("result", []):
                    quotes.append({
                        "symbol": quote.get("symbol"),
                        "name": quote.get("shortName", quote.get("longName", "")),
                        "price": quote.get("regularMarketPrice"),
                        "change": quote.get("regularMarketChange"),
                        "change_pct": quote.get("regularMarketChangePercent"),
                        "volume": quote.get("regularMarketVolume"),
                        "market_cap": quote.get("marketCap"),
                        "pe_ratio": quote.get("trailingPE"),
                        "52w_high": quote.get("fiftyTwoWeekHigh"),
                        "52w_low": quote.get("fiftyTwoWeekLow"),
                        "avg_volume": quote.get("averageDailyVolume3Month"),
                        "category": "stock",
                        "scanned_at": datetime.now(timezone.utc).isoformat(),
                    })

                logger.info(f"WebScanner: fetched {len(quotes)} stock quotes")

        except Exception as e:
            logger.warning(f"WebScanner: stock scan failed: {e}")

        return quotes

    # ─── Crypto Scanner ─────────────────────────────────────

    async def scan_crypto(self, symbols: list[str] | None = None) -> list[dict]:
        """Fetch crypto prices from CoinGecko API (free, no key)."""
        config = _load_intelligence_config()
        symbols = symbols or config.get("crypto_symbols", ["BTC", "ETH"])
        coins = []

        try:
            # Map common symbols to CoinGecko IDs (from YAML)
            symbol_map = config.get("crypto_id_map", {
                "BTC": "bitcoin", "ETH": "ethereum", "SOL": "solana",
            })
            ids = ",".join(symbol_map.get(s, s.lower()) for s in symbols)
            url = f"https://api.coingecko.com/api/v3/simple/price?ids={ids}&vs_currencies=usd&include_24hr_change=true&include_market_cap=true&include_24hr_vol=true"

            resp = await self._client.get(url)
            if resp.status_code == 200:
                data = resp.json()
                for coin_id, info in data.items():
                    coins.append({
                        "symbol": next((k for k, v in symbol_map.items() if v == coin_id), coin_id.upper()),
                        "name": coin_id.replace("-", " ").title(),
                        "price": info.get("usd"),
                        "change_24h_pct": info.get("usd_24h_change"),
                        "market_cap": info.get("usd_market_cap"),
                        "volume_24h": info.get("usd_24h_vol"),
                        "category": "crypto",
                        "scanned_at": datetime.now(timezone.utc).isoformat(),
                    })

                logger.info(f"WebScanner: fetched {len(coins)} crypto prices")

        except Exception as e:
            logger.warning(f"WebScanner: crypto scan failed: {e}")

        return coins

    # ─── ChromaDB Storage ───────────────────────────────────

    async def store_in_memory(self, items: list[dict], collection_name: str = "web_intelligence") -> int:
        """Store scanned items in ChromaDB for agent context injection."""
        if not items:
            return 0

        try:
            chroma = self._get_chroma()
            collection = chroma.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )

            documents = []
            metadatas = []
            ids = []

            for item in items:
                # Create a searchable document from the item
                doc_text = f"{item.get('title', item.get('symbol', ''))} - {item.get('description', item.get('name', ''))}"
                doc_id = hashlib.md5(f"{doc_text[:200]}".encode()).hexdigest()

                documents.append(doc_text[:1000])
                metadatas.append({k: str(v)[:500] for k, v in item.items() if v is not None})
                ids.append(doc_id)

            # Upsert to avoid duplicates
            collection.upsert(
                documents=documents,
                metadatas=metadatas,
                ids=ids,
            )

            logger.info(f"WebScanner: stored {len(items)} items in '{collection_name}'")
            return len(items)

        except Exception as e:
            logger.warning(f"WebScanner: ChromaDB storage failed: {e}")
            return 0

    # ─── Full Scan Cycle ────────────────────────────────────

    async def full_scan(self) -> dict:
        """Run a full scan cycle across all sources."""
        result = {"articles": 0, "stocks": 0, "crypto": 0, "errors": []}

        # 1. RSS Feeds — all categories from YAML
        all_feeds = _load_feeds()
        for category in all_feeds:
            try:
                articles = await self.scan_rss(category)
                stored = await self.store_in_memory(articles, f"web_{category}")
                result["articles"] += stored
            except Exception as e:
                result["errors"].append(f"RSS {category}: {e}")

        # 2. Stock Market
        try:
            stocks = await self.scan_stocks()
            stored = await self.store_in_memory(stocks, "web_stock_market")
            result["stocks"] = stored
        except Exception as e:
            result["errors"].append(f"Stocks: {e}")

        # 3. Crypto
        try:
            crypto = await self.scan_crypto()
            stored = await self.store_in_memory(crypto, "web_crypto")
            result["crypto"] = stored
        except Exception as e:
            result["errors"].append(f"Crypto: {e}")

        logger.info(
            f"WebScanner: full scan complete — "
            f"articles={result['articles']}, stocks={result['stocks']}, crypto={result['crypto']}, "
            f"errors={len(result['errors'])}"
        )
        return result

    # ─── Context for Agents ─────────────────────────────────

    async def get_context(self, query: str, collection_name: str = "web_intelligence", n_results: int = 5) -> list[dict]:
        """Search scanned data for agent context injection."""
        try:
            chroma = self._get_chroma()
            collection = chroma.get_or_create_collection(name=collection_name)

            if collection.count() == 0:
                return []

            results = collection.query(
                query_texts=[query],
                n_results=min(n_results, collection.count()),
            )

            items = []
            if results and results["metadatas"] and results["metadatas"][0]:
                for meta in results["metadatas"][0]:
                    items.append(dict(meta))

            return items

        except Exception as e:
            logger.warning(f"WebScanner: context query failed: {e}")
            return []

    # ─── Helpers ────────────────────────────────────────────

    @staticmethod
    def _get_text(element, tag: str) -> Optional[str]:
        """Get text content from an XML element."""
        el = element.find(tag)
        return el.text.strip() if el is not None and el.text else None

    @staticmethod
    def _get_attr(element, tag: str, attr: str) -> Optional[str]:
        """Get attribute from an XML element."""
        el = element.find(tag)
        return el.get(attr) if el is not None else None

    async def close(self):
        """Close the HTTP client."""
        await self._client.aclose()


# ─── Singleton ──────────────────────────────────────────────
_scanner: WebScanner | None = None


def get_web_scanner() -> WebScanner:
    global _scanner
    if _scanner is None:
        _scanner = WebScanner()
    return _scanner
