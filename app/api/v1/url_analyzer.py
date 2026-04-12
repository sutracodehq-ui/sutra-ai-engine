"""URL Analyzer API endpoints — serves digital footprint data to the marketing tool UI."""

import asyncio

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc, func
from pydantic import BaseModel
from typing import Optional

from app.dependencies import get_db, get_current_tenant
from app.lib.llm_pipeline import get_pipeline_config
from app.lib.response import ok, fail
from app.models.url_analysis import UrlAnalysis

router = APIRouter(prefix="/url-analyzer", tags=["url-analyzer"])


# ─── Request/Response Schemas ────────────────────────────────

class AnalyzeRequest(BaseModel):
    url: str
    max_pages: int = 5


class AnalysisSummary(BaseModel):
    id: int
    url: str
    domain: str
    seo_score: int
    security_score: int
    indexed_pages: int
    tech_stack: list
    created_at: str

    class Config:
        from_attributes = True


# ─── Endpoints ───────────────────────────────────────────────

@router.post("/analyze", response_model=dict, summary="Analyze a URL")
async def analyze_url(
    payload: AnalyzeRequest,
    db: AsyncSession = Depends(get_db),
    tenant=Depends(get_current_tenant),
):
    """
    Run a full digital footprint analysis on any URL.

    **What gets scanned:**
    - SEO: titles, meta descriptions, headings, canonical tags, image alt text
    - Google Indexing: estimated indexed pages via `site:domain.com` query
    - robots.txt: crawl directives and sitemap references
    - sitemap.xml: URL count and structure
    - Tech Stack: React, WordPress, Shopify, GA4, GTM, Facebook Pixel, etc.
    - Social Profiles: Facebook, Twitter/X, Instagram, LinkedIn, YouTube, TikTok
    - Structured Data: JSON-LD Schema.org markup
    - Security Headers: HSTS, CSP, X-Frame-Options, etc.
    - Contact Info: emails and phone numbers

    Results are persisted for trend tracking and self-learning. Set `max_pages` (1-10) to control crawl depth.
    """
    timeout = get_pipeline_config("url_analyze").get("crawl_timeout_seconds", 45)
    try:
        scraper = WebScraperService()
        scraped_data = await asyncio.wait_for(
            scraper.analyze_url(payload.url, max_pages=payload.max_pages),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        return fail(message=f"URL analysis timed out for {payload.url}", code=504)

    # Persist for training
    from urllib.parse import urlparse

    analysis = UrlAnalysis(
        tenant_id=tenant.id,
        url=payload.url,
        domain=urlparse(payload.url).netloc,
        seo_score=scraped_data.get("overall_seo_health", {}).get("score", 0),
        security_score=scraped_data.get("security_headers", {}).get("score", 0),
        scraped_data=scraped_data,
        indexed_pages=scraped_data.get("google_indexing", {}).get("indexed_pages_estimate", 0),
        tech_stack=scraped_data.get("tech_stack", []),
    )

    db.add(analysis)
    await db.commit()
    await db.refresh(analysis)

    return ok(
        data={
            "id": analysis.id,
            "url": payload.url,
            "domain": analysis.domain,
            "seo": scraped_data.get("overall_seo_health", {}),
            "google_indexing": scraped_data.get("google_indexing", {}),
            "robots_txt": scraped_data.get("robots_txt", {}),
            "sitemap": scraped_data.get("sitemap", {}),
            "tech_stack": scraped_data.get("tech_stack", []),
            "social_profiles": scraped_data.get("social_profiles", {}),
            "structured_data": scraped_data.get("structured_data", []),
            "security_headers": scraped_data.get("security_headers", {}),
            "emails": scraped_data.get("emails", []),
            "phones": scraped_data.get("phones", []),
            "pages": scraped_data.get("pages", []),
        },
        message="URL analyzed",
    )


@router.get("/analyses", response_model=dict, summary="List Past Analyses")
async def list_analyses(
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    domain: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
    tenant=Depends(get_current_tenant),
):
    """
    List all past URL analyses for this tenant (paginated).

    Use this to populate a history table in the UI showing SEO score trends over time.
    Filter by domain to see all scans for a specific website.
    """
    query = select(UrlAnalysis).where(UrlAnalysis.tenant_id == tenant.id)

    if domain:
        query = query.where(UrlAnalysis.domain.ilike(f"%{domain}%"))

    # Count total
    count_query = select(func.count()).select_from(query.subquery())
    total = (await db.execute(count_query)).scalar() or 0

    # Paginate
    query = query.order_by(desc(UrlAnalysis.created_at)).offset((page - 1) * per_page).limit(per_page)
    result = await db.execute(query)
    analyses = result.scalars().all()

    return {
        "data": [
            {
                "id": a.id,
                "url": a.url,
                "domain": a.domain,
                "seo_score": a.seo_score,
                "security_score": a.security_score,
                "indexed_pages": a.indexed_pages,
                "tech_stack": a.tech_stack,
                "created_at": str(a.created_at),
            }
            for a in analyses
        ],
        "meta": {
            "total": total,
            "page": page,
            "per_page": per_page,
            "last_page": (total + per_page - 1) // per_page,
        },
    }


@router.get("/analyses/{analysis_id}", response_model=dict, summary="Get Analysis Details")
async def get_analysis(
    analysis_id: int,
    db: AsyncSession = Depends(get_db),
    tenant=Depends(get_current_tenant),
):
    """
    Get the full details of a specific URL analysis.

    Includes the complete `scraped_data` JSON blob (all pages, tech stack, social profiles, etc.)
    and the `ai_report` (the LLM-generated natural language audit report).
    """
    stmt = select(UrlAnalysis).where(
        UrlAnalysis.id == analysis_id,
        UrlAnalysis.tenant_id == tenant.id,
    )
    result = await db.execute(stmt)
    analysis = result.scalar_one_or_none()

    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")

    return {
        "id": analysis.id,
        "url": analysis.url,
        "domain": analysis.domain,
        "seo_score": analysis.seo_score,
        "security_score": analysis.security_score,
        "indexed_pages": analysis.indexed_pages,
        "tech_stack": analysis.tech_stack,
        "scraped_data": analysis.scraped_data,
        "ai_report": analysis.ai_report,
        "created_at": str(analysis.created_at),
    }


@router.get("/analyses/{analysis_id}/compare", response_model=dict, summary="Compare With Previous Scan")
async def compare_with_previous(
    analysis_id: int,
    db: AsyncSession = Depends(get_db),
    tenant=Depends(get_current_tenant),
):
    """
    Compare a URL analysis with the most recent previous scan of the same domain.

    Returns `changes` with delta values for SEO score, security score, and indexed pages.
    Use this to power "score trend" indicators (↑ +5, ↓ -3) in the UI.
    """
    # Get current analysis
    stmt = select(UrlAnalysis).where(
        UrlAnalysis.id == analysis_id,
        UrlAnalysis.tenant_id == tenant.id,
    )
    current = (await db.execute(stmt)).scalar_one_or_none()
    if not current:
        raise HTTPException(status_code=404, detail="Analysis not found")

    # Find previous analysis for the same domain
    prev_stmt = (
        select(UrlAnalysis)
        .where(
            UrlAnalysis.domain == current.domain,
            UrlAnalysis.tenant_id == tenant.id,
            UrlAnalysis.id < current.id,
        )
        .order_by(desc(UrlAnalysis.created_at))
        .limit(1)
    )
    previous = (await db.execute(prev_stmt)).scalar_one_or_none()

    comparison = {
        "current": {
            "id": current.id,
            "seo_score": current.seo_score,
            "security_score": current.security_score,
            "indexed_pages": current.indexed_pages,
            "created_at": str(current.created_at),
        },
        "previous": None,
        "changes": {},
    }

    if previous:
        comparison["previous"] = {
            "id": previous.id,
            "seo_score": previous.seo_score,
            "security_score": previous.security_score,
            "indexed_pages": previous.indexed_pages,
            "created_at": str(previous.created_at),
        }
        comparison["changes"] = {
            "seo_score": current.seo_score - previous.seo_score,
            "security_score": current.security_score - previous.security_score,
            "indexed_pages": current.indexed_pages - previous.indexed_pages,
        }

    return comparison
