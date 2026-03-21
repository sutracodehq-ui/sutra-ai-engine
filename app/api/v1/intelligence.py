"""Intelligence routes — Sentiments, Languages, and Brand Analysis."""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, HttpUrl

from app.dependencies import DbSession, get_current_tenant
from app.models.tenant import Tenant
from app.services.intelligence.sentiment import SentimentService
from app.services.intelligence.language import LanguageService
from app.services.rag.brand_extractor import BrandExtractor

router = APIRouter(prefix="/intelligence", tags=["intelligence"])


class TextRequest(BaseModel):
    text: str


class BrandAnalyzeRequest(BaseModel):
    url: str  # We use str because HttpUrl can be too strict for simple domains


@router.post("/sentiment")
async def analyze_sentiment(
    body: TextRequest,
    tenant: Tenant = Depends(get_current_tenant),
):
    """Detect tone and sentiment of text."""
    return await SentimentService.analyze(body.text)


@router.post("/language")
async def detect_language(
    body: TextRequest,
    tenant: Tenant = Depends(get_current_tenant),
):
    """Detect language of text."""
    return await LanguageService.detect(body.text)


@router.post("/brand-analyze")
async def analyze_brand(
    body: BrandAnalyzeRequest,
    tenant: Tenant = Depends(get_current_tenant),
):
    """
    Crawl a URL and extract brand identity.
    Returns Mission, Voice, Tone, Values, etc.
    """
    profile = await BrandExtractor.analyze_url(body.url)
    if not profile:
        raise HTTPException(422, detail=f"Could not analyze brand from {body.url}")
    return profile
