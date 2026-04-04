"""
Video Driver — abstract base class for all video generation providers.

Software Factory pattern: add a new provider = subclass + YAML config.
Mirrors the ImageDriver pattern exactly.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class VideoResponse:
    """Standardised response from any video generation driver."""

    video_url: str
    duration: str                        # "5" or "10"
    provider: str                        # "fal", "runway", etc.
    model: str                           # "fal-ai/kling-video/v2.1/..."
    aspect_ratio: str = "16:9"
    revised_prompt: str | None = None    # Enhanced prompt used
    thumbnail_url: str | None = None     # Preview image (if available)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "video_url": self.video_url,
            "duration": self.duration,
            "provider": self.provider,
            "model": self.model,
            "aspect_ratio": self.aspect_ratio,
            "revised_prompt": self.revised_prompt,
            "thumbnail_url": self.thumbnail_url,
            "metadata": self.metadata,
        }


class VideoDriver(ABC):
    """
    Abstract video generation driver.

    Every provider (Fal/Kling, Runway, Veo) implements this contract.
    The VideoGenerationService uses the driver chain from YAML config
    and calls generate() on each until one succeeds.
    """

    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> VideoResponse:
        """
        Generate a video from a text prompt.

        Args:
            prompt: Enhanced text description of the video
            **kwargs: Provider-specific overrides (duration, aspect_ratio, etc.)

        Returns:
            VideoResponse with temporary video URL and metadata
        """
        ...
