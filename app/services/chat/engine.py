"""
Chat Engine — High-performance entry point for all AI interactions.

Orchestrates the parallel fetching of context and the unified execution pipeline.
"""

import asyncio
import logging
from typing import Any, AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.tenant import Tenant
from app.models.voice_profile import VoiceProfile
from app.models.ai_conversation import AiConversation
from app.services.tenant_service import TenantService
from app.services.chat.aggregator import ContextAggregator
from app.services.chat.pruner import ContextPruner

from app.services.chat.pipeline import ChatPipeline

logger = logging.getLogger(__name__)


class ChatEngine:
    """
    Unified entry point for Chat and Agent execution.
    Handles parallel context gathering and pipeline execution.
    """

    @staticmethod
    async def execute(
        db: AsyncSession,
        tenant: Tenant,
        prompt: str,
        conversation_id: int | None = None,
        voice_profile_id: int | None = None,
        voice_profile_name: str | None = None,
        stream: bool = False,
        **kwargs
    ) -> Any | AsyncGenerator[str, None]:
        """Execute the high-performance chat pipeline."""
        pipeline = ChatPipeline(db, tenant)
        return await pipeline.run(
            prompt,
            conversation_id=conversation_id,
            voice_profile_id=voice_profile_id,
            voice_profile_name=voice_profile_name,
            stream=stream,
            **kwargs
        )

    @staticmethod
    async def get_context(
        db: AsyncSession,
        tenant: Tenant,
        conversation_id: int | None = None,
        voice_profile_id: int | None = None,
        voice_profile_name: str | None = None,
    ) -> dict[str, Any]:
        """
        Fetch all necessary context in parallel.
        - Tenant configuration overrides
        - Voice Profile (by ID or Name)
        - Conversation history (if thread ID provided)
        - (Future) RAG results
        """
        return await ContextAggregator.gather(
            db,
            tenant,
            conversation_id=conversation_id,
            voice_profile_id=voice_profile_id,
            voice_profile_name=voice_profile_name,
        )
