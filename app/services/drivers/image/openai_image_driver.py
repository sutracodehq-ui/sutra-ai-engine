"""
OpenAI DALL-E Image Driver — Premium quality fallback.

Uses the OpenAI images.generate() API for DALL-E 3.
Falls back to this when Fal.ai is unavailable or when
premium quality is explicitly requested.

All model/config values are read from intelligence_config.yaml.
"""

import logging

from openai import AsyncOpenAI

from app.config import get_settings
from app.services.drivers.image.base import ImageDriver, ImageResponse
from app.services.intelligence.config_loader import get_intelligence_config

logger = logging.getLogger(__name__)


class OpenAiImageDriver(ImageDriver):
    """OpenAI DALL-E driver. Config-driven via intelligence_config.yaml."""

    def __init__(self):
        settings = get_settings()
        config = (
            get_intelligence_config()
            .get("image_generation", {})
            .get("providers", {})
            .get("openai", {})
        )
        self._client = AsyncOpenAI(
            api_key=settings.openai_api_key,
            base_url="https://api.openai.com/v1",
        )
        self._model = config.get("model", "dall-e-3")
        self._supports_quality = config.get("supports_quality", True)
        self._supports_style = config.get("supports_style", True)

    def name(self) -> str:
        return "openai"

    async def generate(
        self,
        prompt: str,
        size: str = "1024x1024",
        quality: str = "standard",
        style: str = "natural",
        n: int = 1,
        **options,
    ) -> ImageResponse:
        """Generate image via OpenAI DALL-E API."""
        kwargs = {
            "model": self._model,
            "prompt": prompt,
            "size": size,
            "n": n,
        }

        # Add optional params only if provider supports them
        if self._supports_quality:
            kwargs["quality"] = quality
        if self._supports_style:
            kwargs["style"] = style

        response = await self._client.images.generate(**kwargs)

        image_data = response.data[0]
        logger.info(
            f"🎨 DALL-E image generated: model={self._model}, size={size}, quality={quality}"
        )

        return ImageResponse(
            image_url=image_data.url or "",
            revised_prompt=image_data.revised_prompt or prompt,
            provider="openai",
            model=self._model,
            size=size,
            metadata={"quality": quality, "style": style},
        )
