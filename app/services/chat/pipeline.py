"""
Chat Pipeline Runner — orchestrates the high-performance AI execution flow.
"""

import asyncio
import logging
from typing import Any, AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.tenant import Tenant
from app.services.chat.aggregator import ContextAggregator
from app.services.chat.pruner import ContextPruner
from app.services.intelligence.prompt_cache import get_prompt_cache
from app.services.intelligence.smart_router import SmartRouter
from app.services.llm_service import get_llm_service

logger = logging.getLogger(__name__)


class ChatPipeline:
    """Execution pipeline for synchronous and streaming chat."""

    def __init__(self, db: AsyncSession, tenant: Tenant):
        self.db = db
        self.tenant = tenant
        self.aggregator = ContextAggregator()
        self.pruner = ContextPruner()

    async def run(
        self,
        prompt: str,
        conversation_id: int | None = None,
        voice_profile_id: int | None = None,
        voice_profile_name: str | None = None,
        stream: bool = False,
        **kwargs
    ) -> Any | AsyncGenerator[str, None]:
        """
        Execute the full chat pipeline.
        1. Aggregate Context (Parallel)
        2. Prune/Message Conversion
        3. Prompt Cache Lookup
        4. Smart Routing
        5. LLM Execution (Sync or Stream)
        """
        # Step 1: Parallel Context Gathering
        context = await self.aggregator.gather(
            self.db,
            self.tenant,
            prompt=prompt,
            conversation_id=conversation_id,
            voice_profile_id=voice_profile_id,
            voice_profile_name=voice_profile_name
        )
        
        voice_profile = context["voice_profile"]
        history = context["history"]
        sentiment = context["sentiment"]
        language = context["language"]

        # Step 2: Build System Prompt & Messages
        system_prompt = self._build_system_prompt(voice_profile, sentiment, language)
        messages = self.pruner.compress_for_prompt(history)
        
        # Step 3: Cache Lookup (only if skipping stream or during sync)
        cache = get_prompt_cache()
        cache_key = None
        if not stream:
            # We hash the whole context for caching efficiency
            cache_key = cache.generate_key(self.tenant.id, prompt, system_prompt, messages)
            cached_result = await cache.get(cache_key)
            if cached_result:
                logger.info("⚡ Prompt cache hit!")
                return cached_result

        # Step 4: Smart Routing (Speculative Decoding / Model Selection)
        route = SmartRouter.route(prompt, context=context)
        driver = route["driver"]
        model = route["model"]

        # Step 5: Execution
        service = get_llm_service()
        if stream:
            return service.stream(
                prompt,
                system_prompt=system_prompt,
                messages=messages,
                driver=driver,
                model=model,
                **kwargs
            )
        else:
            result = await service.complete(
                prompt,
                system_prompt=system_prompt,
                messages=messages,
                driver=driver,
                model=model,
                **kwargs
            )
            
            # Save to cache if enabled
            if cache_key:
                await cache.set(cache_key, result)
            
            return result

    def _build_system_prompt(self, voice_profile: Any | None) -> str:
        """Construct the system prompt with voice profile instructions."""
        base = "You are a helpful AI assistant."
        if voice_profile:
            # Call the model's helper method (added previously)
            modifier = voice_profile.to_system_prompt_modifier()
            if modifier:
                return f"{base}\n\n[VOICE INSTRUCTIONS]\n{modifier}"
        return base
