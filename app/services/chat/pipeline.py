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
        # Step 0: Input Safety & Privacy (Shield-AI)
        from app.services.intelligence.moderation import ModerationService
        from app.services.intelligence.pii_redactor import PIIRedactor
        
        # 0.1 Moderation check on raw input
        mod_result = await ModerationService.check(prompt)
        if mod_result["flagged"]:
            logger.warning(f"🚨 Input Safety Violation: {mod_result['categories']}")
            raise ValueError(f"Content safety violation: {', '.join(mod_result['categories'])}")

        # 0.2 Mask PII before processing
        safe_prompt = PIIRedactor.redact(prompt)

        # Step 1: Parallel Context Gathering
        context = await self.aggregator.gather(
            self.db,
            self.tenant,
            prompt=safe_prompt,
            conversation_id=conversation_id,
            voice_profile_id=voice_profile_id,
            voice_profile_name=voice_profile_name
        )
        
        voice_profile = context["voice_profile"]
        history = context["history"]
        sentiment = context["sentiment"]
        language = context["language"]

        # Step 1.5: Knowledge Retrieval (RAG)
        from app.services.rag.knowledge_base import KnowledgeBaseService
        kb = KnowledgeBaseService()
        relevant_chunks = await kb.query(self.tenant.id, safe_prompt)
        context["kb_chunks"] = relevant_chunks

        # 0.3 Sentiment-based Alerts (frustrated user)
        if sentiment["label"] in ["angry", "frustrated"] and self.tenant.webhook_url:
            from app.workers.webhook_job import trigger_frustration_alert
            trigger_frustration_alert.delay(self.tenant.id, sentiment, self.tenant.webhook_url)

        # Step 2: Build System Prompt & Messages
        system_prompt = self._build_system_prompt(voice_profile, sentiment, language, context=context)
        messages = self.pruner.compress_for_prompt(history)
        
        # Step 3: Cache Lookup (only if skipping stream or during sync)
        cache = get_prompt_cache()
        cache_key = None
        if not stream:
            cache_key = cache.generate_key(self.tenant.id, safe_prompt, system_prompt, messages)
            cached_result = await cache.get(cache_key)
            if cached_result:
                logger.info("⚡ Prompt cache hit!")
                return cached_result

        # Step 4: Smart Routing
        router = SmartRouter()
        agent_type = context.get("agent_type", "general") if context else "general"
        route = router.route(safe_prompt, agent_type=agent_type)
        driver = route["driver"]
        model = route["model"]

        # Step 5: Execution
        service = get_llm_service()
        if stream:
            # For stream, we yield chunks; safety happens post-stream or via tokenizer (advanced)
            return service.stream(
                safe_prompt,
                system_prompt=system_prompt,
                messages=messages,
                driver=driver,
                model=model,
                **kwargs
            )
        else:
            result = await service.complete(
                safe_prompt,
                system_prompt=system_prompt,
                messages=messages,
                driver=driver,
                model=model,
                **kwargs
            )
            
            # Step 6: Output Safety & Brand Protection
            from app.services.intelligence.competitor_lock import CompetitorLock
            
            # 6.1 Check output moderation
            out_mod = await ModerationService.check(result.content or "")
            if out_mod["flagged"]:
                logger.error(f"🚨 Output Safety Violation: {out_mod['categories']}")
                result.content = "I apologize, but I cannot generate that content as it violates my safety policy."
            
            # 6.2 Competitor Lock (if profile exists)
            if voice_profile and hasattr(voice_profile, 'metadata'):
                competitors = voice_profile.metadata.get("competitors", [])
                if competitors:
                    result.content = CompetitorLock.apply_guardrail(result.content, competitors)

            # Save to cache if enabled
            if cache_key:
                await cache.set(cache_key, result.to_dict())
            
            return result

    def _build_system_prompt(self, voice_profile: Any | None, sentiment: dict, language: dict, context: dict = None) -> str:
        """Construct the system prompt with brand, linguistic, and knowledge awareness."""
        base = "You are a helpful AI assistant."
        
        # 1. Knowledge Base (RAG)
        if context and context.get("kb_chunks"):
            chunks = context["kb_chunks"]
            base += f"\n\n[KNOWLEDGE BASE]\nUse the following facts to answer the user query if relevant:\n"
            for i, chunk in enumerate(chunks):
                base += f"- {chunk}\n"
        
        # 2. Voice Profile
        if voice_profile:
            modifier = voice_profile.to_system_prompt_modifier()
            if modifier:
                base += f"\n\n[VOICE INSTRUCTIONS]\n{modifier}"

        # 2. Sentiment Awareness
        if sentiment["score"] < -0.4:
            base += "\n\n[TONE ADVICE]\nThe user seems frustrated or angry. Be extra empathetic and concise."
        elif sentiment["label"] == "excited":
            base += "\n\n[TONE ADVICE]\nThe user is excited! Match their energy and be enthusiastic."

        # 3. Native Language Support
        lang_name = language.get("name", "English")
        base += f"\n\n[LANGUAGE]\nThe user is speaking in {lang_name}. Respond fluently in the SAME language unless they ask otherwise."

        return base
