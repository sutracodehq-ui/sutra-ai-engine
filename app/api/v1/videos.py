"""
Video Generation API — config-driven multi-provider video generation.

Routes:
  POST /v1/videos/generate — Generate video from text prompt

Driver chain and all behavior is read from intelligence_config.yaml.
"""

import logging

from fastapi import APIRouter

from app.dependencies import CurrentTenant, DbSession
from app.lib.response import ok, fail
from app.schemas.video import VideoGenerateRequest
from app.services.video.generator import get_video_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/videos", tags=["videos"])


@router.post("/generate", summary="Generate Video from Text")
async def generate_video(
    body: VideoGenerateRequest,
    tenant: CurrentTenant,
    db: DbSession,
):
    """
    Generate an AI video from a text description.

    **Two-phase pipeline:**
    1. Prompt enhancement — text LLM refines the description for cinematic results
    2. Video generation — driver chain generates the video (fal/kling)

    **Returns:**
    - `video_url`: Temporary URL to the generated video (download promptly)
    - `revised_prompt`: The enhanced prompt actually used for generation
    - `provider`: Which provider generated the video (fal)
    - `model`: Specific model used
    - `duration`: Video duration in seconds
    - `aspect_ratio`: Aspect ratio used
    - `thumbnail_url`: Preview thumbnail (if available)
    - `metadata`: Provider-specific metadata (width, height, etc.)

    **⚠️ Note:** Video generation takes 30-120 seconds. Set client timeout accordingly.
    """
    try:
        service = get_video_service()

        # Build context from tenant + request metadata
        context = body.metadata or {}
        context["tenant_name"] = tenant.name
        context["tenant_slug"] = tenant.slug

        result = await service.generate(
            prompt=body.prompt,
            tenant_id=tenant.id,
            context=context,
            duration=body.duration,
            aspect_ratio=body.aspect_ratio,
            provider=body.provider,
        )

        return ok(
            data=result.to_dict(),
            message="Video generated successfully",
        )

    except ValueError as e:
        return fail(message=str(e), code=400)
    except RuntimeError as e:
        logger.error(f"Video generation failed for tenant {tenant.slug}: {e}")
        return fail(message=str(e), code=502)
    except Exception as e:
        logger.exception(f"Unexpected error in video generation: {e}")
        return fail(message="Video generation failed", code=500)
