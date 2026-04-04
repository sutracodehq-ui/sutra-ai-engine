"""
Image Generation Service — config-driven multi-provider image generation.

Software Factory pattern:
- Reads EVERYTHING from intelligence_config.yaml
- Adding a new provider = add YAML entry + driver class. Zero service changes.
- Prompt enhancement pipeline uses existing LLM infrastructure.
- Driver fallback chain ensures high availability.

This file should NEVER be modified to add new providers or change behavior.
All customization happens in YAML config.
"""

import json
import logging
from typing import Any

from app.services.drivers.image.base import ImageDriver, ImageResponse
from app.services.drivers.image.fal_image_driver import FalImageDriver
from app.services.drivers.image.openai_image_driver import OpenAiImageDriver
from app.services.intelligence.config_loader import get_intelligence_config

logger = logging.getLogger(__name__)

# ─── Driver Registry ─────────────────────────────────────────
# Maps config name → class. Adding a new driver = 1 import + 1 line here.
_DRIVER_REGISTRY: dict[str, type[ImageDriver]] = {
    "fal": FalImageDriver,
    "openai": OpenAiImageDriver,
    # Future: "gemini": GeminiImageDriver,
    # Future: "stability": StabilityImageDriver,
}

# Singleton cache for driver instances
_driver_cache: dict[str, ImageDriver] = {}


def _get_driver(name: str) -> ImageDriver:
    """Get or create a cached driver instance by config name."""
    if name not in _driver_cache:
        driver_cls = _DRIVER_REGISTRY.get(name)
        if not driver_cls:
            raise ValueError(f"Unknown image driver: {name}. Available: {list(_DRIVER_REGISTRY.keys())}")
        _driver_cache[name] = driver_cls()
    return _driver_cache[name]


def _get_config() -> dict:
    """Get the image generation config from intelligence_config.yaml."""
    return get_intelligence_config().get("image_generation", {})


class ImageGenerationService:
    """
    Generic image generation engine. Reads EVERYTHING from YAML config.

    Two-phase generation:
    1. Prompt Enhancement — uses text LLM to refine the user's brief description
    2. Image Generation — uses the driver chain (first success wins)

    Adding a new provider = add YAML entry + driver class. Zero changes here.
    """

    async def generate(
        self,
        prompt: str,
        tenant_id: int | None = None,
        context: dict | None = None,
        **overrides,
    ) -> ImageResponse:
        """
        Generate image with automatic prompt enhancement and provider fallback.

        Args:
            prompt: User's text description
            tenant_id: Tenant ID for brand context injection
            context: Additional context (brand name, colors, etc.)
            **overrides: Override defaults (size, quality, style, n, provider)

        Returns:
            ImageResponse with image URL and metadata
        """
        config = _get_config()
        if not config.get("enabled"):
            raise ValueError("Image generation is disabled in configuration")

        # ─── Resolve None → YAML defaults (schema passes None for all unset fields) ───
        defaults = config.get("defaults", {})
        for key in ("size", "quality", "style", "n"):
            if overrides.get(key) is None:
                overrides[key] = defaults.get(key)

        # ─── Phase 1: Prompt Enhancement (text LLM) ─────────
        enhanced_prompt = await self._enhance_prompt(prompt, tenant_id, context, config)

        # ─── Resolve size preset if provided ─────────────────
        size = overrides.get("size", "1024x1024")
        size_presets = config.get("size_presets", {})
        if size in size_presets:
            size = size_presets[size]
        overrides["size"] = size

        # ─── Phase 2: Generate via driver chain ──────────────
        # If specific provider requested, try it directly
        requested_provider = overrides.pop("provider", None)
        if requested_provider:
            driver = _get_driver(requested_provider)
            result = await driver.generate(enhanced_prompt, **overrides)
            result.revised_prompt = result.revised_prompt or enhanced_prompt
            return result

        # Otherwise, fallback through the chain
        chain = config.get("driver_chain", ["fal", "openai"])
        defaults = config.get("defaults", {})
        merged = {**defaults, **overrides}
        # Remove keys that aren't generation params
        merged.pop("response_format", None)

        last_error = None
        for driver_name in chain:
            try:
                driver = _get_driver(driver_name)
                result = await driver.generate(enhanced_prompt, **merged)
                result.revised_prompt = result.revised_prompt or enhanced_prompt
                logger.info(f"✅ Image generated via {driver_name}: {result.image_url[:80]}...")
                return result
            except Exception as e:
                last_error = e
                logger.warning(f"⚠️ Image driver '{driver_name}' failed: {e}")
                continue

        raise RuntimeError(f"All image drivers failed. Last error: {last_error}")

    async def _enhance_prompt(
        self,
        prompt: str,
        tenant_id: int | None,
        context: dict | None,
        config: dict,
    ) -> str:
        """
        Use text LLM to enhance the prompt before generation.
        Uses existing SmartRouter/pipeline infrastructure — zero new code.
        """
        enhancement_config = config.get("prompt_enhancement", {})
        if not enhancement_config.get("enabled"):
            return prompt

        try:
            from app.services.llm_service import get_llm_service

            llm = get_llm_service()

            # Build brand context from available data
            brand_context = ""
            if context:
                brand_parts = []
                brand_name = context.get("tenant_name") or context.get("brand_name", "")
                if brand_name:
                    brand_parts.append(f"Brand: {brand_name}")
                brand_colors = context.get("brand_colors", "")
                if brand_colors:
                    brand_parts.append(f"Colors: {brand_colors}")
                industry = context.get("tenant_industry") or context.get("industry", "")
                if industry:
                    brand_parts.append(f"Industry: {industry}")
                brand_context = "; ".join(brand_parts)

            system_prompt = (
                "You are an expert image prompt engineer. Transform the user's brief description "
                "into a detailed, professional-grade image generation prompt.\n\n"
                "Rules:\n"
                "1. Add specific visual details: lighting, composition, color palette, art style\n"
                "2. Include brand identity if context is provided\n"
                "3. Add quality modifiers for photorealistic output\n"
                "4. Keep the enhanced prompt under 400 characters\n"
                "5. Output ONLY the enhanced prompt text. No JSON, no explanations."
            )

            user_prompt = f"Enhance this image prompt: \"{prompt}\""
            if brand_context:
                user_prompt += f"\n\nBrand context: {brand_context}"

            response = await llm.complete(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.7,
                max_tokens=500,
            )

            enhanced = response.content.strip().strip('"').strip("'")
            if enhanced and len(enhanced) > 20:
                logger.info(f"🔧 Prompt enhanced: '{prompt[:50]}...' → '{enhanced[:50]}...'")
                return enhanced

            return prompt

        except Exception as e:
            logger.warning(f"Prompt enhancement failed, using original: {e}")
            return prompt


# ─── Module-level singleton ──────────────────────────────────

_service_instance: ImageGenerationService | None = None


def get_image_service() -> ImageGenerationService:
    """Get or create the ImageGenerationService singleton."""
    global _service_instance
    if _service_instance is None:
        _service_instance = ImageGenerationService()
    return _service_instance
