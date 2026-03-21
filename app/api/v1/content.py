"""
Content Generation API — dedicated endpoints for the marketing tool.

These wrap the agent hub but provide structured input/output
specific to each content type. Every response goes through ApiResponse.
"""

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from typing import Optional

from app.dependencies import CurrentTenant, DbSession
from app.middleware.response import ApiResponse
from app.services.agents.hub import get_agent_hub

router = APIRouter(prefix="/content", tags=["content"])


# ─── Request Schemas ─────────────────────────────────────────────

class SocialPostRequest(BaseModel):
    platform: str = Field(..., description="Target platform: instagram, twitter, linkedin, facebook, tiktok")
    topic: str = Field(..., description="What the post is about")
    tone: str = Field("professional", description="Tone of voice: professional, casual, witty, bold")
    include_hashtags: bool = Field(True)
    include_image_prompt: bool = Field(True)
    brand_voice: Optional[str] = Field(None, description="Brand voice description for consistency")


class EmailTemplateRequest(BaseModel):
    purpose: str = Field(..., description="Email purpose: welcome, promo, newsletter, re-engage, announcement")
    topic: str = Field(..., description="What the email is about")
    audience: str = Field("general", description="Target audience description")
    tone: str = Field("professional", description="Tone: professional, friendly, urgent, empathetic")
    include_subject_variants: bool = Field(True)


class AdCopyRequest(BaseModel):
    platform: str = Field(..., description="Ad platform: google, meta, linkedin, twitter")
    product: str = Field(..., description="Product/service being advertised")
    objective: str = Field("conversions", description="Campaign objective: awareness, traffic, conversions, leads")
    target_audience: str = Field(..., description="Who you're targeting")
    num_variants: int = Field(3, ge=1, le=5, description="Number of A/B variants to generate")


class RepurposeRequest(BaseModel):
    content: str = Field(..., description="Original content to repurpose")
    source_format: str = Field("blog", description="Source format: blog, social, email, video-script, ad")
    target_formats: list[str] = Field(..., description="Target formats: blog, social, email, ad, sms, whatsapp")
    brand_voice: Optional[str] = Field(None)


class CalendarSuggestRequest(BaseModel):
    industry: str = Field(..., description="Industry/niche")
    goals: list[str] = Field(..., description="Marketing goals: awareness, engagement, leads, sales")
    timeframe: str = Field("week", description="Planning timeframe: week, biweekly, month")
    platforms: list[str] = Field(default=["instagram", "linkedin"], description="Target platforms")
    brand_voice: Optional[str] = Field(None)


class LandingPageRequest(BaseModel):
    product: str = Field(..., description="Product/service name")
    value_proposition: str = Field(..., description="Core value proposition")
    target_audience: str = Field(..., description="Who this page is for")
    sections: list[str] = Field(
        default=["hero", "features", "testimonials", "cta"],
        description="Page sections to generate"
    )
    tone: str = Field("professional", description="Tone of voice")


# ─── Endpoints ───────────────────────────────────────────────────

@router.post("/social-post", summary="Generate Social Post")
async def generate_social_post(
    body: SocialPostRequest,
    tenant: CurrentTenant,
    db: DbSession,
):
    """
    Generate a platform-specific social media post.

    **Returns structured JSON:**
    - `caption`: Full post caption
    - `hashtags`: Relevant hashtags (if requested)
    - `cta`: Call-to-action suggestion
    - `image_prompt`: AI image generation prompt (if requested)
    - `optimal_posting_time`: Best time to post for this platform
    """
    hub = get_agent_hub()
    prompt = f"""Generate a {body.platform} social media post about: {body.topic}
Tone: {body.tone}
{"Brand voice: " + body.brand_voice if body.brand_voice else ""}
{"Include relevant hashtags." if body.include_hashtags else "No hashtags."}
{"Include an AI image generation prompt." if body.include_image_prompt else ""}
Respond in JSON with keys: caption, hashtags, cta, image_prompt, optimal_posting_time"""

    context = {"tenant_slug": tenant.slug}
    response = await hub.run("social", prompt, context, db=db)
    return ApiResponse.ok(data={"content": response.content, "agent": "social", "platform": body.platform})


@router.post("/email-template", summary="Generate Email Template")
async def generate_email_template(
    body: EmailTemplateRequest,
    tenant: CurrentTenant,
    db: DbSession,
):
    """
    Generate a complete email template with subject line variants.

    **Returns:**
    - `subject_variants`: 3 subject line options
    - `preheader`: Preview text
    - `body_html`: Full email body
    - `cta`: Call-to-action text and button copy
    """
    hub = get_agent_hub()
    prompt = f"""Generate a {body.purpose} email about: {body.topic}
Target audience: {body.audience}
Tone: {body.tone}
{"Generate 3 subject line variants." if body.include_subject_variants else ""}
Respond in JSON with keys: subject_variants, preheader, body_html, cta_text, cta_button"""

    context = {"tenant_slug": tenant.slug}
    response = await hub.run("email_campaign", prompt, context, db=db)
    return ApiResponse.ok(data={"content": response.content, "agent": "email_campaign", "purpose": body.purpose})


@router.post("/ad-copy", summary="Generate Ad Copy")
async def generate_ad_copy(
    body: AdCopyRequest,
    tenant: CurrentTenant,
    db: DbSession,
):
    """
    Generate ad copy with A/B testing variants.

    **Returns per variant:**
    - `headline`: Ad headline (platform-appropriate length)
    - `description`: Ad description/body
    - `cta`: Call-to-action
    - `display_url`: Suggested display URL path
    """
    hub = get_agent_hub()
    prompt = f"""Generate {body.num_variants} ad copy variants for {body.platform}.
Product: {body.product}
Campaign objective: {body.objective}
Target audience: {body.target_audience}
Format each variant with: headline, description, cta, display_url
Respond in JSON with key: variants (array)"""

    context = {"tenant_slug": tenant.slug}
    response = await hub.run("ad_creative", prompt, context, db=db)
    return ApiResponse.ok(data={"content": response.content, "agent": "ad_creative", "platform": body.platform, "variants_requested": body.num_variants})


@router.post("/repurpose", summary="Repurpose Content")
async def repurpose_content(
    body: RepurposeRequest,
    tenant: CurrentTenant,
    db: DbSession,
):
    """
    Transform one piece of content into multiple formats.

    Example: Blog post → Social posts + Email + Ad copy + SMS
    """
    hub = get_agent_hub()
    targets = ", ".join(body.target_formats)
    prompt = f"""Repurpose this {body.source_format} content into these formats: {targets}
{"Brand voice: " + body.brand_voice if body.brand_voice else ""}

ORIGINAL CONTENT:
{body.content}

Respond in JSON with a key for each target format."""

    context = {"tenant_slug": tenant.slug}
    response = await hub.run("content_repurpose", prompt, context, db=db)
    return ApiResponse.ok(data={"content": response.content, "agent": "content_repurpose", "source": body.source_format, "targets": body.target_formats})


@router.post("/calendar-suggest", summary="Suggest Content Calendar")
async def suggest_content_calendar(
    body: CalendarSuggestRequest,
    tenant: CurrentTenant,
    db: DbSession,
):
    """
    Generate a content calendar with topics and optimal timing.

    **Returns:**
    - Array of content suggestions, each with:
      `date`, `platform`, `topic`, `content_type`, `optimal_time`, `goal_alignment`
    """
    hub = get_agent_hub()
    goals = ", ".join(body.goals)
    platforms = ", ".join(body.platforms)
    prompt = f"""Create a {body.timeframe} content calendar for:
Industry: {body.industry}
Goals: {goals}
Platforms: {platforms}
{"Brand voice: " + body.brand_voice if body.brand_voice else ""}
For each entry include: date, platform, topic, content_type, optimal_time, goal_alignment
Respond in JSON with key: calendar (array)"""

    context = {"tenant_slug": tenant.slug}
    response = await hub.run("campaign_strategist", prompt, context, db=db)
    return ApiResponse.ok(data={"content": response.content, "agent": "campaign_strategist", "timeframe": body.timeframe})


@router.post("/landing-page", summary="Generate Landing Page Copy")
async def generate_landing_page(
    body: LandingPageRequest,
    tenant: CurrentTenant,
    db: DbSession,
):
    """
    Generate full landing page copy from a brief.

    **Returns sections:**
    - `hero`: Headline + subheadline + CTA
    - `features`: Feature blocks with titles and descriptions
    - `testimonials`: Social proof quotes
    - `cta`: Final call-to-action block
    """
    hub = get_agent_hub()
    sections = ", ".join(body.sections)
    prompt = f"""Generate landing page copy for:
Product: {body.product}
Value proposition: {body.value_proposition}
Target audience: {body.target_audience}
Tone: {body.tone}
Sections to generate: {sections}
Respond in JSON with a key for each section."""

    context = {"tenant_slug": tenant.slug}
    response = await hub.run("landing_page_builder", prompt, context, db=db)
    return ApiResponse.ok(data={"content": response.content, "agent": "landing_page_builder", "product": body.product})
