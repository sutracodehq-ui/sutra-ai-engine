"""
Video Generation Service — config-driven multi-provider video generation.

Software Factory pattern:
- Reads EVERYTHING from intelligence_config.yaml
- Adding a new provider = add YAML entry + driver class. Zero service changes.
- Prompt enhancement uses existing LLM infrastructure.
- Driver fallback chain ensures high availability.

This file should NEVER be modified to add new providers or change behavior.
"""

import logging
from typing import Any

from app.services.drivers.video.base import VideoDriver, VideoResponse
from app.services.drivers.video.fal_video_driver import FalVideoDriver
from app.services.intelligence.config_loader import get_intelligence_config

logger = logging.getLogger(__name__)

# ─── Driver Registry ─────────────────────────────────────────
# Maps config name → class. Adding a new driver = 1 import + 1 line here.
_DRIVER_REGISTRY: dict[str, type[VideoDriver]] = {
    "fal": FalVideoDriver,
    # Future: "runway": RunwayVideoDriver,
    # Future: "veo": VeoVideoDriver,
}

# Singleton cache for driver instances
_driver_cache: dict[str, VideoDriver] = {}


def _get_driver(name: str) -> VideoDriver:
    """Get or create a cached driver instance by config name."""
    if name not in _driver_cache:
        driver_cls = _DRIVER_REGISTRY.get(name)
        if not driver_cls:
            raise ValueError(f"Unknown video driver: {name}. Available: {list(_DRIVER_REGISTRY.keys())}")
        _driver_cache[name] = driver_cls()
    return _driver_cache[name]


def _get_config() -> dict:
    """Get the video generation config from intelligence_config.yaml."""
    return get_intelligence_config().get("video_generation", {})


class VideoGenerationService:
    """
    Generic video generation engine. Reads EVERYTHING from YAML config.

    Two-phase generation:
    1. Prompt Enhancement — uses text LLM to refine the user's brief description
    2. Video Generation — uses the driver chain (first success wins)

    Adding a new provider = add YAML entry + driver class. Zero changes here.
    """

    async def generate(
        self,
        prompt: str,
        tenant_id: int | None = None,
        context: dict | None = None,
        **overrides,
    ) -> VideoResponse:
        """
        Generate video with automatic prompt enhancement and provider fallback.

        Args:
            prompt: User's text description
            tenant_id: Tenant ID for brand context injection
            context: Additional context (brand name, colors, etc.)
            **overrides: Override defaults (duration, aspect_ratio, provider)

        Returns:
            VideoResponse with video URL and metadata
        """
        config = _get_config()
        if not config.get("enabled"):
            raise ValueError("Video generation is disabled in configuration")

        # ─── Resolve None → YAML defaults ─────────────────────
        defaults = config.get("defaults", {})
        for key in ("duration", "aspect_ratio"):
            if overrides.get(key) is None:
                overrides[key] = defaults.get(key)

        # ─── Enforce max duration from safety config ──────────
        safety = config.get("safety", {})
        max_duration = safety.get("max_duration")
        if max_duration and overrides.get("duration"):
            if int(overrides["duration"]) > int(max_duration):
                overrides["duration"] = max_duration

        # ─── Phase 1: Prompt Enhancement (text LLM) ──────────
        enhanced_prompt = await self._enhance_prompt(prompt, tenant_id, context, config)

        # ─── Phase 2: Generate via driver chain ───────────────
        requested_provider = overrides.pop("provider", None)
        if requested_provider:
            driver = _get_driver(requested_provider)
            result = await driver.generate(enhanced_prompt, **overrides)
            result.revised_prompt = result.revised_prompt or enhanced_prompt
            return result

        # Fallback through the chain
        chain = config.get("driver_chain", ["fal"])
        merged = {**defaults, **overrides}

        last_error = None
        for driver_name in chain:
            try:
                driver = _get_driver(driver_name)
                result = await driver.generate(enhanced_prompt, **merged)
                result.revised_prompt = result.revised_prompt or enhanced_prompt
                logger.info(f"✅ Video generated via {driver_name}: {result.video_url[:80]}...")
                return result
            except Exception as e:
                last_error = e
                logger.warning(f"⚠️ Video driver '{driver_name}' failed: {e}")
                continue

        raise RuntimeError(f"All video drivers failed. Last error: {last_error}")

    async def _enhance_prompt(
        self,
        prompt: str,
        tenant_id: int | None,
        context: dict | None,
        config: dict,
    ) -> str:
        """
        Use text LLM to enhance the prompt for cinematic video generation.
        Uses existing SmartRouter/pipeline infrastructure — zero new code.
        """
        enhancement_config = config.get("prompt_enhancement", {})
        if not enhancement_config.get("enabled"):
            return prompt

        try:
            from app.services.llm_service import get_llm_service

            llm = get_llm_service()

            # Build brand context
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
                "You are an expert video director and prompt engineer. Transform the user's brief "
                "description into a detailed, cinematic video generation prompt.\n\n"
                "Rules:\n"
                "1. Add specific camera movements: pan, zoom, tracking shot, dolly, aerial\n"
                "2. Describe lighting: golden hour, studio, neon, natural, dramatic\n"
                "3. Include motion and action details for dynamic footage\n"
                "4. Apply brand identity if context is provided\n"
                "5. Add mood, atmosphere, and color grading notes\n"
                "6. Keep the enhanced prompt under 500 characters\n"
                "7. Output ONLY the enhanced prompt text. No JSON, no explanations."
            )

            user_prompt = f'Enhance this video prompt: "{prompt}"'
            if brand_context:
                user_prompt += f"\n\nBrand context: {brand_context}"

            response = await llm.complete(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.7,
                max_tokens=600,
            )

            enhanced = response.content.strip().strip('"').strip("'")
            if enhanced and len(enhanced) > 20:
                logger.info(f"🎬 Video prompt enhanced: '{prompt[:50]}...' → '{enhanced[:50]}...'")
                return enhanced

            return prompt

        except Exception as e:
            logger.warning(f"Video prompt enhancement failed, using original: {e}")
            return prompt


# ─── Module-level singleton ──────────────────────────────────

_service_instance: VideoGenerationService | None = None


def get_video_service() -> VideoGenerationService:
    """Get or create the VideoGenerationService singleton."""
    global _service_instance
    if _service_instance is None:
        _service_instance = VideoGenerationService()
    return _service_instance
