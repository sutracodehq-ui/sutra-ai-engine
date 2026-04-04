"""
Abstract Image Driver — the contract every image provider must implement.

Software Factory pattern: every image driver is a standardized, interchangeable
production unit. Swap Fal.ai → OpenAI → Stability with zero consumer changes.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ImageResponse:
    """Standardized response from any image generation provider."""

    image_url: str = ""            # URL to generated image (temporary)
    image_b64: str = ""            # Base64 encoded image (if requested)
    revised_prompt: str = ""       # Provider's revised prompt (DALL-E returns this)
    provider: str = ""             # Driver name (e.g., 'fal', 'openai')
    model: str = ""                # Model used (e.g., 'fal-ai/flux/schnell')
    size: str = ""                 # Size used for generation
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize to dict for API responses."""
        result = {
            "image_url": self.image_url,
            "revised_prompt": self.revised_prompt,
            "provider": self.provider,
            "model": self.model,
            "size": self.size,
        }
        if self.image_b64:
            result["image_b64"] = self.image_b64
        if self.metadata:
            result["metadata"] = self.metadata
        return result


class ImageDriver(ABC):
    """
    Abstract base for all image generation drivers.

    Every driver implements 2 methods:
    - name(): driver identifier
    - generate(): text prompt → ImageResponse
    """

    @abstractmethod
    def name(self) -> str:
        """Driver identifier (e.g., 'fal', 'openai')."""
        ...

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        size: str = "1024x1024",
        quality: str = "standard",
        style: str = "natural",
        n: int = 1,
        **options,
    ) -> ImageResponse:
        """Generate image(s) from a text prompt."""
        ...
