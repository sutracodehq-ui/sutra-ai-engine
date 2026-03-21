"""Intelligence routes — Kernel intelligence services + marketing analysis endpoints."""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

from app.dependencies import DbSession, get_current_tenant
from app.models.tenant import Tenant
from app.middleware.response import ApiResponse
from app.services.intelligence.sentiment import SentimentService
from app.services.intelligence.language import LanguageService
from app.services.rag.brand_extractor import BrandExtractor
from app.services.intelligence.web_scraper import WebScraperService
from app.services.agents.hub import get_agent_hub

router = APIRouter(prefix="/intelligence", tags=["intelligence"])


# ─── Request Schemas ─────────────────────────────────────────────

class TextRequest(BaseModel):
    text: str


class BrandAnalyzeRequest(BaseModel):
    url: str


class UrlAnalyzeRequest(BaseModel):
    url: str = Field(..., description="URL to analyze")
    max_pages: int = Field(5, ge=1, le=20, description="Max pages to crawl")


class SeoAuditRequest(BaseModel):
    url: str = Field(..., description="URL to audit")
    max_pages: int = Field(5, ge=1, le=20)
    include_recommendations: bool = Field(True, description="Include AI-generated improvement suggestions")


class HashtagSuggestRequest(BaseModel):
    topic: str = Field(..., description="Topic or keyword to generate hashtags for")
    industry: str = Field("general", description="Industry/niche for context")
    platform: str = Field("instagram", description="Target platform: instagram, twitter, linkedin, tiktok")
    count: int = Field(20, ge=5, le=50, description="Number of hashtags to suggest")


class CompetitorAnalyzeRequest(BaseModel):
    url: str = Field(..., description="Competitor's website URL")
    focus_areas: list[str] = Field(
        default=["positioning", "content_strategy", "social_presence", "seo"],
        description="Areas to analyze"
    )


# ─── Existing Kernel Endpoints ───────────────────────────────────

@router.post("/sentiment", summary="Analyze Sentiment")
async def analyze_sentiment(
    body: TextRequest,
    tenant: Tenant = Depends(get_current_tenant),
):
    """
    Detect tone, sentiment score, and emotional vibe of text.

    **Returns:** `score` (-1 to 1), `label` (positive/neutral/negative), `vibe` (excitement/calm/anger/etc.)
    """
    result = await SentimentService.analyze(body.text)
    return ApiResponse.ok(data=result)


@router.post("/language", summary="Detect Language")
async def detect_language(
    body: TextRequest,
    tenant: Tenant = Depends(get_current_tenant),
):
    """
    Detect the language of input text with confidence score.

    **Returns:** `language` (ISO code), `name` (English name), `confidence` (0-1)
    """
    result = await LanguageService.detect(body.text)
    return ApiResponse.ok(data=result)


@router.post("/brand-analyze", summary="Analyze Brand from URL")
async def analyze_brand(
    body: BrandAnalyzeRequest,
    tenant: Tenant = Depends(get_current_tenant),
):
    """
    Crawl a URL and extract brand identity.

    **Returns:** Mission, Voice, Tone, Values, Colors, Industry classification.
    """
    profile = await BrandExtractor.analyze_url(body.url)
    if not profile:
        return ApiResponse.error(
            message=f"Could not analyze brand from {body.url}",
            code="BRAND_ANALYSIS_FAILED",
            status=422,
        )
    return ApiResponse.ok(data=profile)


# ─── New Marketing Intelligence Endpoints ────────────────────────

@router.post("/url-analyze", summary="Analyze URL Digital Footprint")
async def analyze_url(
    body: UrlAnalyzeRequest,
    tenant: Tenant = Depends(get_current_tenant),
    db: DbSession = None,
):
    """
    Crawl any URL and extract its complete digital footprint.

    **Returns:**
    - SEO score, meta tags, headings, canonical tags
    - Google indexing status (estimated indexed pages)
    - Tech stack detection (React, WordPress, GA4, etc.)
    - Social media profiles
    - Structured data (Schema.org JSON-LD)
    - Security headers score
    - robots.txt and sitemap.xml analysis
    """
    scraper = WebScraperService()
    scraped_data = await scraper.analyze_url(body.url, max_pages=body.max_pages)

    return ApiResponse.ok(data=scraped_data, meta={"url": body.url, "pages_crawled": len(scraped_data.get("pages", []))})


@router.post("/seo-audit", summary="Full SEO Audit")
async def seo_audit(
    body: SeoAuditRequest,
    tenant: Tenant = Depends(get_current_tenant),
    db: DbSession = None,
):
    """
    Run a comprehensive SEO audit on any URL.

    **Checks:**
    - Title tags (presence + length 30-60 chars)
    - Meta descriptions (presence + length 120-160 chars)
    - H1 tags (exactly 1 per page)
    - Canonical URLs
    - Image alt text coverage
    - robots.txt and sitemap.xml
    - Google indexing status
    - Structured data (Schema.org)
    - Security headers
    - Internal vs external link ratio

    If `include_recommendations` is true, also generates AI-powered improvement suggestions.
    """
    scraper = WebScraperService()
    scraped_data = await scraper.analyze_url(body.url, max_pages=body.max_pages)

    audit_result = {
        "url": body.url,
        "seo_health": scraped_data.get("overall_seo_health", {}),
        "google_indexing": scraped_data.get("google_indexing", {}),
        "robots_txt": scraped_data.get("robots_txt", {}),
        "sitemap": scraped_data.get("sitemap", {}),
        "structured_data": scraped_data.get("structured_data", []),
        "security_headers": scraped_data.get("security_headers", {}),
        "pages": scraped_data.get("pages", []),
    }

    # Generate AI recommendations if requested
    if body.include_recommendations:
        import json
        hub = get_agent_hub()
        prompt = f"""Based on this SEO audit data, provide prioritized recommendations:
{json.dumps(audit_result, indent=2, default=str)}

Respond in JSON with key: recommendations (array of objects with: priority (high/medium/low), issue, recommendation, impact)"""
        context = {"tenant_slug": tenant.slug}
        response = await hub.run("seo", prompt, context, db=db)
        audit_result["ai_recommendations"] = response.content

    return ApiResponse.ok(data=audit_result, meta={"pages_audited": len(audit_result["pages"])})


@router.post("/hashtag-suggest", summary="Suggest Hashtags")
async def suggest_hashtags(
    body: HashtagSuggestRequest,
    tenant: Tenant = Depends(get_current_tenant),
    db: DbSession = None,
):
    """
    Generate trending and niche hashtag suggestions for a topic.

    **Returns:**
    - `trending`: High-reach hashtags (large audience, competitive)
    - `niche`: Lower-reach but higher-engagement hashtags
    - `branded`: Brand-specific hashtag suggestions
    - `reach_estimate`: Estimated reach category per hashtag
    """
    hub = get_agent_hub()
    prompt = f"""Suggest {body.count} hashtags for the topic "{body.topic}" on {body.platform}.
Industry: {body.industry}
Categorize into: trending (high reach), niche (high engagement), branded (unique).
For each hashtag include an estimated reach: huge (1M+), large (100K-1M), medium (10K-100K), small (<10K).
Respond in JSON with keys: trending, niche, branded (each an array of objects with: hashtag, reach_estimate)"""

    context = {"tenant_slug": tenant.slug}
    response = await hub.run("social", prompt, context, db=db)
    return ApiResponse.ok(data={"content": response.content, "platform": body.platform, "topic": body.topic})


@router.post("/competitor-analyze", summary="Analyze Competitor")
async def analyze_competitor(
    body: CompetitorAnalyzeRequest,
    tenant: Tenant = Depends(get_current_tenant),
    db: DbSession = None,
):
    """
    Deep-analyze a competitor's digital presence.

    **Returns:**
    - `positioning`: How they position their brand
    - `content_strategy`: Content types, posting frequency, themes
    - `social_presence`: Active social platforms and follower estimates
    - `seo_analysis`: SEO score, indexed pages, tech stack
    - `strengths`: What they do well
    - `weaknesses`: Gaps you can exploit
    - `opportunities`: Actionable competitive advantages
    """
    # Step 1: Scrape competitor's site
    scraper = WebScraperService()
    scraped_data = await scraper.analyze_url(body.url, max_pages=5)

    # Step 2: Feed scrape to competitor analyst agent
    import json
    hub = get_agent_hub()
    focus = ", ".join(body.focus_areas)
    prompt = f"""Analyze this competitor's digital footprint and provide detailed competitive intelligence.
Focus areas: {focus}

SCRAPED DATA:
{json.dumps(scraped_data, indent=2, default=str)}

Respond in JSON with keys: positioning, content_strategy, social_presence, strengths, weaknesses, opportunities"""

    context = {"tenant_slug": tenant.slug}
    response = await hub.run("competitor_analyst", prompt, context, db=db)

    return ApiResponse.ok(
        data={"content": response.content, "scraped_summary": {
            "domain": scraped_data.get("domain", ""),
            "tech_stack": scraped_data.get("tech_stack", []),
            "social_profiles": scraped_data.get("social_profiles", {}),
            "seo_score": scraped_data.get("overall_seo_health", {}).get("score", 0),
        }},
        meta={"competitor_url": body.url}
    )
