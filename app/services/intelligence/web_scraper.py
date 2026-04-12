"""
WebScraperService — Extracts digital footprint from any URL.

Crawls a website and extracts:
- Page titles, meta descriptions, headings
- Social media links, contact info
- Technology stack (frameworks, analytics, CMS)
- Content structure and keyword density
- Image alt tags, broken links
"""

import logging
import re
from typing import Dict, List, Any, Optional
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class WebScraperService:
    """Service to crawl and extract structured data from websites."""

    # Common tech signatures in HTML
    TECH_SIGNATURES = {
        "React": ["react", "_next", "__NEXT_DATA__"],
        "Vue.js": ["vue", "__vue__"],
        "Angular": ["ng-app", "ng-version"],
        "WordPress": ["wp-content", "wp-includes"],
        "Shopify": ["cdn.shopify.com", "myshopify"],
        "Wix": ["wix.com", "wixsite"],
        "Google Analytics": ["google-analytics.com", "gtag", "UA-", "G-"],
        "Google Tag Manager": ["googletagmanager.com", "GTM-"],
        "Facebook Pixel": ["fbq(", "connect.facebook.net"],
        "HubSpot": ["hs-analytics", "hubspot"],
        "Hotjar": ["hotjar", "hj("],
        "Intercom": ["intercom", "intercomSettings"],
        "Drift": ["drift.com", "driftt"],
        "Tailwind CSS": ["tailwindcss", "tw-"],
        "Bootstrap": ["bootstrap"],
    }

    SOCIAL_PATTERNS = {
        "facebook": r"facebook\.com/[\w.]+",
        "twitter": r"(?:twitter|x)\.com/[\w]+",
        "instagram": r"instagram\.com/[\w.]+",
        "linkedin": r"linkedin\.com/(?:company|in)/[\w-]+",
        "youtube": r"youtube\.com/(?:channel|c|@)[\w-]+",
        "tiktok": r"tiktok\.com/@[\w.]+",
        "pinterest": r"pinterest\.com/[\w]+",
        "github": r"github\.com/[\w-]+",
    }

    def __init__(self):
        self._client = httpx.AsyncClient(
            timeout=15.0,
            follow_redirects=True,
            headers={
                "User-Agent": "SutraAI-Analyzer/1.0 (Digital Footprint Scanner)"
            }
        )

    async def analyze_url(self, url: str, max_pages: int = 5) -> Dict[str, Any]:
        """
        Full digital footprint analysis of a URL.
        Crawls the main page + up to max_pages internal links.
        """
        if not url.startswith("http"):
            url = f"https://{url}"

        result = {
            "url": url,
            "domain": urlparse(url).netloc,
            "pages": [],
            "social_profiles": {},
            "tech_stack": [],
            "emails": [],
            "phones": [],
            "overall_seo_health": {},
            "google_indexing": {},
            "robots_txt": {},
            "sitemap": {},
            "structured_data": [],
            "security_headers": {},
        }

        try:
            # 1. Crawl main page
            main_page = await self._scrape_page(url)
            result["pages"].append(main_page)

            # 2. Find internal links and crawl them
            internal_links = self._extract_internal_links(main_page.get("raw_html", ""), url)
            crawled = {url}
            
            for link in internal_links[:max_pages - 1]:
                if link not in crawled:
                    crawled.add(link)
                    try:
                        page_data = await self._scrape_page(link)
                        result["pages"].append(page_data)
                    except Exception as e:
                        logger.warning(f"Failed to crawl {link}: {e}")

            # 3. Aggregate findings
            all_html = " ".join(p.get("raw_html", "") for p in result["pages"])
            result["social_profiles"] = self._extract_social_profiles(all_html)
            result["tech_stack"] = self._detect_tech_stack(all_html)
            result["emails"] = self._extract_emails(all_html)
            result["phones"] = self._extract_phones(all_html)
            result["overall_seo_health"] = self._compute_seo_health(result["pages"])

            # 4. Google Indexing & SEO Deep Scan
            domain = urlparse(url).netloc
            result["google_indexing"] = await self._check_google_indexing(domain)
            result["robots_txt"] = await self._check_robots_txt(url)
            result["sitemap"] = await self._check_sitemap(url)
            result["structured_data"] = self._extract_structured_data(main_page.get("raw_html", ""))
            result["security_headers"] = await self._check_security_headers(url)

            # Clean raw HTML from response
            for page in result["pages"]:
                page.pop("raw_html", None)

        except Exception as e:
            logger.error(f"WebScraper: Failed to analyze {url}: {e}")
            result["error"] = str(e)

        return result

    async def _scrape_page(self, url: str) -> Dict[str, Any]:
        """Scrape a single page and extract structured data."""
        response = await self._client.get(url)
        html = response.text
        soup = BeautifulSoup(html, "html.parser")

        # Meta tags
        title = soup.title.string.strip() if soup.title and soup.title.string else ""
        meta_desc = ""
        meta_tag = soup.find("meta", attrs={"name": "description"})
        if meta_tag:
            meta_desc = meta_tag.get("content", "")

        # Canonical tag
        canonical = ""
        canonical_tag = soup.find("link", attrs={"rel": "canonical"})
        if canonical_tag:
            canonical = canonical_tag.get("href", "")

        # Meta robots
        meta_robots = ""
        robots_tag = soup.find("meta", attrs={"name": "robots"})
        if robots_tag:
            meta_robots = robots_tag.get("content", "")

        # OG tags
        og_tags = {}
        for tag in soup.find_all("meta", attrs={"property": re.compile(r"^og:")}):
            og_tags[tag.get("property", "")] = tag.get("content", "")

        # Twitter Card tags
        twitter_tags = {}
        for tag in soup.find_all("meta", attrs={"name": re.compile(r"^twitter:")}):
            twitter_tags[tag.get("name", "")] = tag.get("content", "")

        # Headings
        headings = {
            "h1": [h.get_text(strip=True) for h in soup.find_all("h1")],
            "h2": [h.get_text(strip=True) for h in soup.find_all("h2")],
            "h3": [h.get_text(strip=True) for h in soup.find_all("h3")],
        }

        # Images
        images = []
        for img in soup.find_all("img")[:20]:
            images.append({
                "src": img.get("src", ""),
                "alt": img.get("alt", ""),
                "has_alt": bool(img.get("alt"))
            })

        # Internal vs External links
        internal_count = 0
        external_count = 0
        base_domain = urlparse(url).netloc
        for a in soup.find_all("a", href=True):
            href = a["href"]
            full = urljoin(url, href)
            if urlparse(full).netloc == base_domain:
                internal_count += 1
            else:
                external_count += 1

        # Word count
        body_text = soup.get_text(separator=" ", strip=True)
        word_count = len(body_text.split())

        return {
            "url": url,
            "status_code": response.status_code,
            "title": title,
            "title_length": len(title),
            "meta_description": meta_desc,
            "meta_description_length": len(meta_desc),
            "canonical": canonical,
            "meta_robots": meta_robots,
            "og_tags": og_tags,
            "twitter_tags": twitter_tags,
            "headings": headings,
            "word_count": word_count,
            "image_count": len(images),
            "images_missing_alt": sum(1 for img in images if not img["has_alt"]),
            "internal_links": internal_count,
            "external_links": external_count,
            "raw_html": html,
        }

    # ─── Google Indexing & SEO Deep Scan ─────────────────────────

    async def _check_google_indexing(self, domain: str) -> Dict[str, Any]:
        """
        Check Google Search for current indexing status.
        Uses a `site:domain.com` query to estimate indexed page count.
        """
        result = {
            "indexed_pages_estimate": 0,
            "top_indexed_pages": [],
            "indexing_status": "unknown",
        }

        try:
            # Google search: site:domain.com
            search_url = f"https://www.google.com/search?q=site:{domain}&num=10"
            response = await self._client.get(
                search_url,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                }
            )
            html = response.text
            soup = BeautifulSoup(html, "html.parser")

            # Extract result count ("About X results")
            result_stats = soup.find("div", id="result-stats")
            if result_stats:
                text = result_stats.get_text()
                numbers = re.findall(r"[\d,]+", text)
                if numbers:
                    result["indexed_pages_estimate"] = int(numbers[0].replace(",", ""))

            # Extract top indexed URLs from search results
            for g in soup.find_all("div", class_="g")[:5]:
                link = g.find("a")
                title_el = g.find("h3")
                if link and title_el:
                    result["top_indexed_pages"].append({
                        "url": link.get("href", ""),
                        "title": title_el.get_text(strip=True)
                    })

            if result["indexed_pages_estimate"] > 0:
                result["indexing_status"] = "indexed"
            elif response.status_code == 200:
                result["indexing_status"] = "not_indexed_or_blocked"

        except Exception as e:
            logger.warning(f"Google indexing check failed: {e}")
            result["error"] = str(e)

        return result

    async def _check_robots_txt(self, url: str) -> Dict[str, Any]:
        """Fetch and analyze robots.txt for crawling directives."""
        parsed = urlparse(url)
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
        result = {"exists": False, "content": "", "directives": [], "sitemaps_referenced": []}

        try:
            response = await self._client.get(robots_url)
            if response.status_code == 200:
                content = response.text
                result["exists"] = True
                result["content"] = content[:2000]  # Cap at 2KB

                # Parse directives
                for line in content.splitlines():
                    line = line.strip()
                    if line and not line.startswith("#"):
                        if line.lower().startswith("sitemap:"):
                            result["sitemaps_referenced"].append(line.split(":", 1)[1].strip())
                        elif line.lower().startswith(("user-agent:", "disallow:", "allow:", "crawl-delay:")):
                            result["directives"].append(line)
        except Exception as e:
            logger.warning(f"robots.txt check failed: {e}")

        return result

    async def _check_sitemap(self, url: str) -> Dict[str, Any]:
        """Fetch and analyze sitemap.xml for indexed URLs."""
        parsed = urlparse(url)
        sitemap_url = f"{parsed.scheme}://{parsed.netloc}/sitemap.xml"
        result = {"exists": False, "url_count": 0, "sample_urls": [], "type": "unknown"}

        try:
            response = await self._client.get(sitemap_url)
            if response.status_code == 200 and "xml" in response.headers.get("content-type", ""):
                content = response.text
                result["exists"] = True

                # Count <url> or <sitemap> entries
                url_count = content.count("<url>") or content.count("<loc>")
                result["url_count"] = url_count

                # Detect sitemap index vs urlset
                if "<sitemapindex" in content:
                    result["type"] = "sitemap_index"
                else:
                    result["type"] = "urlset"

                # Extract sample URLs
                soup = BeautifulSoup(content, "xml")
                for loc in soup.find_all("loc")[:10]:
                    result["sample_urls"].append(loc.get_text(strip=True))

        except Exception as e:
            logger.warning(f"sitemap.xml check failed: {e}")

        return result

    def _extract_structured_data(self, html: str) -> List[Dict[str, Any]]:
        """Extract JSON-LD structured data (Schema.org) from the HTML."""
        soup = BeautifulSoup(html, "html.parser")
        schemas = []

        for script in soup.find_all("script", type="application/ld+json"):
            try:
                import json
                data = json.loads(script.string)
                if isinstance(data, list):
                    for item in data:
                        schemas.append({
                            "type": item.get("@type", "Unknown"),
                            "data": {k: v for k, v in item.items() if k != "@context"}
                        })
                elif isinstance(data, dict):
                    schemas.append({
                        "type": data.get("@type", "Unknown"),
                        "data": {k: v for k, v in data.items() if k != "@context"}
                    })
            except Exception:
                pass

        return schemas

    async def _check_security_headers(self, url: str) -> Dict[str, Any]:
        """Check for important security headers."""
        important_headers = [
            "strict-transport-security",
            "content-security-policy",
            "x-content-type-options",
            "x-frame-options",
            "x-xss-protection",
            "referrer-policy",
            "permissions-policy",
        ]

        result = {"score": 0, "present": [], "missing": []}

        try:
            response = await self._client.head(url)
            headers_lower = {k.lower(): v for k, v in response.headers.items()}

            for h in important_headers:
                if h in headers_lower:
                    result["present"].append(h)
                else:
                    result["missing"].append(h)

            result["score"] = int((len(result["present"]) / len(important_headers)) * 100)

        except Exception as e:
            logger.warning(f"Security headers check failed: {e}")

        return result

    # ─── Existing Helper Methods ─────────────────────────────────

    def _extract_internal_links(self, html: str, base_url: str) -> List[str]:
        """Extract internal links from HTML."""
        soup = BeautifulSoup(html, "html.parser")
        base_domain = urlparse(base_url).netloc
        links = set()

        for a in soup.find_all("a", href=True):
            href = a["href"]
            full_url = urljoin(base_url, href)
            parsed = urlparse(full_url)
            
            if parsed.netloc == base_domain and parsed.scheme in ("http", "https"):
                clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
                links.add(clean_url)

        return list(links)

    def _extract_social_profiles(self, html: str) -> Dict[str, List[str]]:
        """Extract social media profile links."""
        profiles = {}
        for platform, pattern in self.SOCIAL_PATTERNS.items():
            matches = re.findall(pattern, html, re.IGNORECASE)
            if matches:
                profiles[platform] = list(set(matches))
        return profiles

    def _detect_tech_stack(self, html: str) -> List[str]:
        """Detect technologies used on the website."""
        detected = []
        html_lower = html.lower()
        
        for tech, signatures in self.TECH_SIGNATURES.items():
            if any(sig.lower() in html_lower for sig in signatures):
                detected.append(tech)
        
        return detected

    def _extract_emails(self, html: str) -> List[str]:
        """Extract email addresses from HTML."""
        pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
        return list(set(re.findall(pattern, html)))[:10]

    def _extract_phones(self, html: str) -> List[str]:
        """Extract phone numbers from HTML."""
        pattern = r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}"
        return list(set(re.findall(pattern, html)))[:10]

    def _compute_seo_health(self, pages: List[Dict]) -> Dict[str, Any]:
        """Compute a comprehensive SEO health score from scraped pages."""
        total = len(pages)
        if total == 0:
            return {"score": 0, "issues": []}

        issues = []
        checks_passed = 0
        total_checks = 0

        # Title checks
        has_title = sum(1 for p in pages if p.get("title"))
        good_title_len = sum(1 for p in pages if 30 <= p.get("title_length", 0) <= 60)
        total_checks += total * 2
        checks_passed += has_title + good_title_len
        if has_title < total:
            issues.append(f"{total - has_title} pages missing title tags")
        if good_title_len < total:
            issues.append(f"{total - good_title_len} pages have title length outside 30-60 chars")

        # Meta description checks
        has_desc = sum(1 for p in pages if p.get("meta_description"))
        good_desc_len = sum(1 for p in pages if 120 <= p.get("meta_description_length", 0) <= 160)
        total_checks += total * 2
        checks_passed += has_desc + good_desc_len
        if has_desc < total:
            issues.append(f"{total - has_desc} pages missing meta descriptions")
        if good_desc_len < total:
            issues.append(f"{total - good_desc_len} pages have meta description outside 120-160 chars")

        # H1 checks
        has_h1 = sum(1 for p in pages if p.get("headings", {}).get("h1"))
        single_h1 = sum(1 for p in pages if len(p.get("headings", {}).get("h1", [])) == 1)
        total_checks += total * 2
        checks_passed += has_h1 + single_h1
        if has_h1 < total:
            issues.append(f"{total - has_h1} pages missing H1 headings")
        if single_h1 < total:
            issues.append(f"{total - single_h1} pages have zero or multiple H1 tags (best practice: exactly 1)")

        # Canonical check
        has_canonical = sum(1 for p in pages if p.get("canonical"))
        total_checks += total
        checks_passed += has_canonical
        if has_canonical < total:
            issues.append(f"{total - has_canonical} pages missing canonical tags")

        # Image alt check
        total_missing_alt = sum(p.get("images_missing_alt", 0) for p in pages)
        if total_missing_alt > 0:
            issues.append(f"{total_missing_alt} images missing alt text across all pages")

        score = int((checks_passed / total_checks) * 100) if total_checks else 0

        return {"score": score, "issues": issues, "pages_analyzed": total}

