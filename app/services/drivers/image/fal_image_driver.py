"""
Fal.ai Image Driver — Flux Schnell ($0.003/image).

Primary image generation driver. Ultra-fast and cost-effective.
Uses the fal-client Python SDK for communication.

All model/config values are read from intelligence_config.yaml.
Adding new Fal.ai models = edit YAML only. Never change this file.
"""

import asyncio
import logging
from functools import partial

import fal_client

from app.config import get_settings
from app.services.drivers.image.base import ImageDriver, ImageResponse
from app.services.intelligence.config_loader import get_intelligence_config

logger = logging.getLogger(__name__)


class FalImageDriver(ImageDriver):
    """Fal.ai Flux driver. Config-driven via intelligence_config.yaml."""

    def __init__(self):
        config = (
            get_intelligence_config()
            .get("image_generation", {})
            .get("providers", {})
            .get("fal", {})
        )
        self._model = config.get("model", "fal-ai/flux/schnell")
        self._steps = config.get("num_inference_steps", 4)

        # Set credentials from env
        settings = get_settings()
        if settings.fal_key:
            import os
            os.environ.setdefault("FAL_KEY", settings.fal_key)

    def name(self) -> str:
        return "fal"

    async def generate(
        self,
        prompt: str,
        size: str = "square_hd",
        quality: str = "standard",
        style: str = "natural",
        n: int = 1,
        **options,
    ) -> ImageResponse:
        """Generate image via Fal.ai Flux model."""
        arguments = {
            "prompt": prompt,
            "image_size": size,
            "num_images": n,
            "num_inference_steps": self._steps,
        }

        # fal_client.subscribe is synchronous — run in thread pool
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            partial(fal_client.subscribe, self._model, arguments=arguments),
        )

        images = result.get("images", [])
        if not images:
            raise RuntimeError("Fal.ai returned no images")

        image_data = images[0]
        logger.info(
            f"🎨 Fal.ai image generated: model={self._model}, "
            f"size={image_data.get('width')}x{image_data.get('height')}"
        )

        return ImageResponse(
            image_url=image_data.get("url", ""),
            provider="fal",
            model=self._model,
            size=size,
            metadata={
                "width": image_data.get("width"),
                "height": image_data.get("height"),
                "content_type": image_data.get("content_type", "image/jpeg"),
            },
        )
