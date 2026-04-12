"""
Fal.ai Kling Video Driver — text-to-video via Kling v2.1.

Uses the same fal-client SDK as FalImageDriver.
Reads all config from intelligence_config.yaml → video_generation.providers.fal.
"""

import asyncio
import logging
from functools import partial

import fal_client

from app.services.drivers.video.base import VideoDriver, VideoResponse
from app.services.intelligence.config_loader import get_intelligence_config

logger = logging.getLogger(__name__)


class FalVideoDriver(VideoDriver):
    """Fal.ai Kling video generation driver."""

    def __init__(self):
        # Set credentials from env
        from app.config import get_settings
        import os
        settings = get_settings()
        if settings.fal_key:
            os.environ.setdefault("FAL_KEY", settings.fal_key)

    def _get_provider_config(self) -> dict:
        """Read provider config from YAML."""
        config = get_intelligence_config().get("video_generation", {})
        return config.get("providers", {}).get("fal", {})

    async def generate(self, prompt: str, **kwargs) -> VideoResponse:
        """
        Generate video via Fal.ai Kling API.

        fal_client.subscribe is synchronous — we wrap in run_in_executor.
        """
        provider_config = self._get_provider_config()
        model = provider_config.get("model", "fal-ai/kling-video/v2.1/standard/text-to-video")

        duration = str(kwargs.get("duration", "5"))
        aspect_ratio = kwargs.get("aspect_ratio", "16:9")

        arguments = {
            "prompt": prompt,
            "duration": duration,
            "aspect_ratio": aspect_ratio,
        }

        logger.info(f"🎬 Fal.ai video generation: model={model}, duration={duration}s, ratio={aspect_ratio}")

        # fal_client.subscribe is blocking — offload to thread pool
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            partial(
                fal_client.subscribe,
                model,
                arguments=arguments,
                with_logs=True,
            ),
        )

        # Extract video URL from response
        video_data = result.get("video", {})
        video_url = video_data.get("url", "")

        if not video_url:
            raise RuntimeError(f"Fal.ai returned no video URL. Response: {result}")

        # Build metadata
        metadata = {
            "content_type": video_data.get("content_type", "video/mp4"),
            "file_size": video_data.get("file_size"),
            "width": video_data.get("width"),
            "height": video_data.get("height"),
            "request_id": result.get("request_id"),
        }

        return VideoResponse(
            video_url=video_url,
            duration=duration,
            provider="fal",
            model=model,
            aspect_ratio=aspect_ratio,
            revised_prompt=prompt,
            thumbnail_url=result.get("thumbnail", {}).get("url"),
            metadata={k: v for k, v in metadata.items() if v is not None},
        )
