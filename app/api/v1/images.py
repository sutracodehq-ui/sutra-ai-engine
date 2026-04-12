"""
Image Generation API — config-driven multi-provider image generation.

Routes:
  POST /v1/images/generate — Generate image from text prompt

Driver chain and all behavior is read from intelligence_config.yaml.
"""

import logging

from fastapi import APIRouter

from app.dependencies import CurrentTenant, DbSession
from app.lib.response import ok, fail
from app.schemas.image import ImageGenerateRequest
from app.services.image.generator import get_image_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/images", tags=["images"])


@router.post("/generate", summary="Generate Image from Text")
async def generate_image(
    body: ImageGenerateRequest,
    tenant: CurrentTenant,
    db: DbSession,
):
    """
    Generate an AI image from a text description.

    **Two-phase pipeline:**
    1. Prompt enhancement — text LLM refines the description for better results
    2. Image generation — driver chain generates the image (fal → openai fallback)

    **Returns:**
    - `image_url`: Temporary URL to the generated image (download promptly)
    - `revised_prompt`: The enhanced prompt actually used for generation
    - `provider`: Which provider generated the image (fal, openai)
    - `model`: Specific model used
    - `size`: Dimensions or preset used
    - `metadata`: Provider-specific metadata (width, height, etc.)
    """
    try:
        service = get_image_service()

        # Build context from tenant + request metadata
        context = body.metadata or {}
        context["tenant_name"] = tenant.name
        context["tenant_slug"] = tenant.slug

        result = await service.generate(
            prompt=body.prompt,
            tenant_id=tenant.id,
            context=context,
            size=body.size,
            quality=body.quality,
            style=body.style,
            n=body.n,
            provider=body.provider,
        )

        return ok(
            data=result.to_dict(),
            message="Image generated successfully",
        )

    except ValueError as e:
        return fail(message=str(e), code=400)
    except RuntimeError as e:
        logger.error(f"Image generation failed for tenant {tenant.slug}: {e}")
        return fail(message=str(e), code=502)
    except Exception as e:
        logger.exception(f"Unexpected error in image generation: {e}")
        return fail(message="Image generation failed", code=500)
