"""
Context Aggregator — fetches all required data for a chat request in parallel.

Uses asyncio.gather to minimize pre-LLM latency.
"""

import asyncio
import logging
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.tenant import Tenant
from app.models.voice_profile import VoiceProfile
from app.models.ai_conversation import AiConversation
from app.models.ai_task import AiTask

logger = logging.getLogger(__name__)


class ContextAggregator:
    """Logic for concurrent data retrieval."""

    @classmethod
    async def gather(
        cls,
        db: AsyncSession,
        tenant: Tenant,
        conversation_id: int | None = None,
        voice_profile_id: int | None = None,
        voice_profile_name: str | None = None,
    ) -> dict[str, Any]:
        """
        Main entry point for gathering context.
        Returns a dict containing:
        - voice_profile: VoiceProfile | None
        - conversation: AiConversation | None
        - history: list[AiTask]
        """
        tasks = []

        # 1. Fetch Voice Profile
        tasks.append(cls._fetch_voice_profile(db, tenant.id, voice_profile_id, voice_profile_name))

        # 2. Fetch Conversation + Recent History
        if conversation_id:
            tasks.append(cls._fetch_conversation_with_history(db, tenant.id, conversation_id))
        else:
            tasks.append(asyncio.sleep(0, result=(None, [])))

        # Run all in parallel
        voice_profile, (conversation, history) = await asyncio.gather(*tasks)

        return {
            "voice_profile": voice_profile,
            "conversation": conversation,
            "history": history,
        }

    @staticmethod
    async def _fetch_voice_profile(
        db: AsyncSession,
        tenant_id: int,
        profile_id: int | None = None,
        profile_name: str | None = None,
    ) -> VoiceProfile | None:
        """Fetch a specific voice profile or the tenant's default."""
        if profile_id:
            result = await db.execute(
                select(VoiceProfile).where(
                    VoiceProfile.id == profile_id,
                    VoiceProfile.tenant_id == tenant_id
                )
            )
            return result.scalar_one_or_none()
        
        if profile_name:
            result = await db.execute(
                select(VoiceProfile).where(
                    VoiceProfile.name == profile_name,
                    VoiceProfile.tenant_id == tenant_id
                )
            )
            profile = result.scalar_one_or_none()
            if profile:
                return profile

        # Fallback to default if nothing specifically requested
        result = await db.execute(
            select(VoiceProfile).where(
                VoiceProfile.tenant_id == tenant_id,
                VoiceProfile.is_default.is_(True)
            )
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def _fetch_conversation_with_history(
        db: AsyncSession,
        tenant_id: int,
        conversation_id: int
    ) -> tuple[AiConversation | None, list[AiTask]]:
        """Fetch a conversation and its recent task history (last 10 turns)."""
        result = await db.execute(
            select(AiConversation).where(
                AiConversation.id == conversation_id,
                AiConversation.tenant_id == tenant_id
            )
        )
        conversation = result.scalar_one_or_none()
        if not conversation:
            return None, []

        # Fetch last 10 successful tasks for context
        history_result = await db.execute(
            select(AiTask).where(
                AiTask.conversation_id == conversation_id,
                AiTask.status == "completed"
            ).order_by(AiTask.created_at.desc()).limit(10)
        )
        history = list(history_result.scalars().all())
        # Reverse to get chronological order
        history.reverse()

        return conversation, history
