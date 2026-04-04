"""
Pydantic schemas for video generation endpoints.

All defaults are resolved at runtime from intelligence_config.yaml.
Schema only validates structure — YAML drives all default values.
"""

from pydantic import BaseModel, Field


class VideoGenerateRequest(BaseModel):
    """POST /v1/videos/generate request body.

    All optional fields resolve their defaults from
    intelligence_config.yaml → video_generation.defaults at runtime.
    """

    prompt: str = Field(..., min_length=1, description="Text description of the video to generate")
    duration: str | None = Field(default=None, description="Video duration in seconds (from YAML config)")
    aspect_ratio: str | None = Field(default=None, description="Aspect ratio (from YAML config)")
    provider: str | None = Field(default=None, description="Force a specific provider. If null, uses YAML driver chain.")
    metadata: dict | None = Field(default_factory=dict, description="Additional context (brand_name, brand_colors, industry)")
