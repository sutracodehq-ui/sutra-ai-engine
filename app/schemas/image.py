"""
Pydantic schemas for image generation endpoints.

All defaults are resolved at runtime from intelligence_config.yaml.
Schema only validates structure — YAML drives all default values.
"""

from pydantic import BaseModel, Field


class ImageGenerateRequest(BaseModel):
    """POST /v1/images/generate request body.

    All optional fields resolve their defaults from
    intelligence_config.yaml → image_generation.defaults at runtime.
    """

    prompt: str = Field(..., min_length=1, description="Text description of the image to generate")
    size: str | None = Field(default=None, description="Image size preset or dimensions (from YAML config)")
    quality: str | None = Field(default=None, description="Image quality level (from YAML config)")
    style: str | None = Field(default=None, description="Image style (from YAML config)")
    n: int | None = Field(default=None, ge=1, description="Number of images to generate (from YAML config)")
    provider: str | None = Field(default=None, description="Force a specific provider. If null, uses YAML driver chain.")
    metadata: dict | None = Field(default_factory=dict, description="Additional context (brand_name, brand_colors, industry)")
