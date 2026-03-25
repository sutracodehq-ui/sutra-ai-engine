"""
Context Aggregator — fetches all necessary data for a chat turn.

Software Factory: Parallel context gathering. 
Fetches Tenant, Voice Profile, Conversation History, Sentiment, 
and Language in a single concurrent batch.
"""

import asyncio
import logging
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.ai_conversation import AiConversation
from app.models.ai_task import AiTask
from app.models.tenant import Tenant
from app.models.voice_profile import VoiceProfile

logger = logging.getLogger(__name__)


class ContextAggregator:
    """Service to fetch chat context in parallel."""

    async def gather(
        self,
        db: AsyncSession,
        tenant: Tenant,
        prompt: str | None = None,
        conversation_id: int | None = None,
        voice_profile_id: int | None = None,
        voice_profile_name: str | None = None
    ) -> dict[str, Any]:
        """Fetch all context in parallel."""
        from app.services.intelligence.guardian import get_guardian
        from app.services.intelligence.language import LanguageService

        tasks = [
            self._fetch_voice_profile(db, tenant.id, voice_profile_id, voice_profile_name),
            self._fetch_history(db, conversation_id) if conversation_id else self._empty_history(),
        ]
        
        # Add intelligence tasks if prompt is provided
        if prompt:
            tasks.append(SentimentService.analyze(prompt))
            tasks.append(LanguageService.detect(prompt))
        else:
            tasks.append(self._empty_sentiment())
            tasks.append(self._empty_language())

        results = await asyncio.gather(*tasks)

        return {
            "tenant": tenant,
            "voice_profile": results[0],
            "history": results[1],
            "sentiment": results[2],
            "language": results[3]
        }

    async def _fetch_voice_profile(
        self, db: AsyncSession, tenant_id: int, vp_id: int | None, vp_name: str | None
    ) -> VoiceProfile | None:
        """Fetch voice profile by ID or Name."""
        if not vp_id and not vp_name:
            return None

        stmt = select(VoiceProfile).where(VoiceProfile.tenant_id == tenant_id)
        if vp_id:
            stmt = stmt.where(VoiceProfile.id == vp_id)
        else:
            stmt = stmt.where(VoiceProfile.name == vp_name)

        result = await db.execute(stmt)
        return result.scalar_one_or_none()

    async def _fetch_history(self, db: AsyncSession, conversation_id: int) -> list[AiTask]:
        """Fetch recent history for a conversation."""
        stmt = (
            select(AiTask)
            .where(AiTask.conversation_id == conversation_id)
            .order_by(AiTask.created_at.desc())
            .limit(10)
        )
        result = await db.execute(stmt)
        tasks = list(result.scalars().all())
        return tasks[::-1]  # Return in chronological order

    async def _empty_history(self) -> list:
        return []

    async def _empty_sentiment(self) -> dict:
        return {"score": 0.0, "label": "neutral", "vibe": "unknown"}

    async def _empty_language(self) -> dict:
        return {"code": "en", "name": "English", "confidence": 1.0}
