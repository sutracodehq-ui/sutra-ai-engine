"""
Chat Pipeline Runner — orchestrates the high-performance AI execution flow.

Updated to use the 4 core engines: Brain, Guardian, Memory, Driver.
"""

import asyncio
import logging
from typing import Any, AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.tenant import Tenant
from app.services.chat.aggregator import ContextAggregator
from app.services.chat.pruner import ContextPruner
from app.services.intelligence.brain import get_brain
from app.services.intelligence.guardian import get_guardian
from app.services.intelligence.memory import get_memory
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
        1. Input Safety (Guardian)
        2. Aggregate Context (Parallel)
        3. Cache Lookup (Memory)
        4. Smart Routing (Brain)
        5. LLM Execution (Sync or Stream)
        6. Output Safety (Guardian)
        """
        brain = get_brain()
        guardian = get_guardian()
        memory = get_memory()

        # Step 0: Input Safety & Privacy (Guardian)
        # 0.1 Moderation check on raw input
        mod_result = await guardian.moderate(prompt)
        if mod_result["flagged"]:
            logger.warning(f"🚨 Input Safety Violation: {mod_result['categories']}")
            raise ValueError(f"Content safety violation: {', '.join(mod_result['categories'])}")

        # 0.2 Mask PII before processing
        safe_prompt = guardian.redact_pii(prompt)

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

        # Step 1.5: Knowledge Retrieval (RAG via Memory)
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
        
        # Step 3: Cache Lookup (Memory)
        cache_key = None
        if not stream:
            cached = await memory.cache_get("chatbot", safe_prompt)
            if cached:
                logger.info("⚡ Cache hit via Memory!")
                from app.services.drivers.base import LlmResponse
                return LlmResponse(content=cached.get("response", ""), total_tokens=0,
                                   driver="cache", model="cached", metadata={"cache": True})

        # Step 4: Smart Routing (Brain)
        agent_type = context.get("agent_type", "general") if context else "general"
        route = brain.route(safe_prompt, agent_type=agent_type)
        driver = route["driver"]
        model = route["model"]
        fallback_chain = route.get("chain")  # full YAML driver chain for resilient fallback

        # Step 5: Execution
        service = get_llm_service()
        if stream:
            return service.stream(
                safe_prompt,
                system_prompt=system_prompt,
                messages=messages,
                driver=driver,
                model=model,
                fallback_chain=fallback_chain,
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
            
            # Step 6: Output Safety & Brand Protection (Guardian)
            # 6.1 Check output moderation
            out_mod = await guardian.moderate(result.content or "")
            if out_mod["flagged"]:
                logger.error(f"🚨 Output Safety Violation: {out_mod['categories']}")
                result.content = "I apologize, but I cannot generate that content as it violates my safety policy."
            
            # 6.2 Competitor Lock (if profile exists)
            if voice_profile and hasattr(voice_profile, 'metadata'):
                competitors = voice_profile.metadata.get("competitors", [])
                if competitors and result.content:
                    for comp in competitors:
                        if comp.lower() in result.content.lower():
                            result.content = result.content.replace(comp, "[COMPETITOR]")

            # Save to cache (Memory)
            if result.content:
                await memory.cache_put("chatbot", safe_prompt, result.content)
            
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
