"""Chat Persistence Service — save/load/delete chat conversations in Postgres.

Software Factory Principle: Single Responsibility.
This service handles ALL DB operations for chat persistence.
The ChatbotEngine and WebSocket handlers call into this — never touch DB directly.
"""

import logging
from datetime import datetime, timezone

from sqlalchemy import delete, select, update, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.chat_session import ChatSession
from app.models.chat_message import ChatMessage

logger = logging.getLogger(__name__)


class ChatPersistenceService:
    """Manages chat session and message persistence in Postgres."""

    # ─── Save a Message ─────────────────────────────────────────

    async def save_message(
        self,
        db: AsyncSession,
        session_id: str,
        tenant_id: int,
        role: str,
        content: str,
        actions: list | None = None,
        confidence: float | None = None,
        channel: str = "websocket",
        language: str | None = None,
    ) -> ChatMessage:
        """
        Save a message to an existing or new session.
        Upserts the session (creates if not exists) and appends the message.
        """
        # 1. Ensure session exists
        session = await self._get_or_create_session(
            db, session_id, tenant_id, channel, language
        )

        # 2. Insert message
        msg = ChatMessage(
            session_id=session_id,
            tenant_id=tenant_id,
            role=role,
            content=content,
            actions=actions,
            confidence=confidence,
        )
        db.add(msg)

        # 3. Update session counters
        session.message_count = (session.message_count or 0) + 1
        session.last_message_at = datetime.now(timezone.utc)
        if language and not session.language:
            session.language = language

        await db.flush()
        logger.debug(f"ChatPersistence: saved {role} message for session {session_id}")
        return msg

    # ─── Load Session History ───────────────────────────────────

    async def get_session_history(
        self,
        db: AsyncSession,
        session_id: str,
        limit: int = 50,
    ) -> list[dict]:
        """
        Load previous messages for a session (ordered by created_at).
        Returns list of dicts ready for frontend consumption.
        """
        result = await db.execute(
            select(ChatMessage)
            .where(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.created_at.asc())
            .limit(limit)
        )
        messages = result.scalars().all()

        return [
            {
                "id": msg.id,
                "role": msg.role,
                "content": msg.content,
                "actions": msg.actions or [],
                "confidence": msg.confidence,
                "created_at": msg.created_at.isoformat() if msg.created_at else None,
            }
            for msg in messages
        ]

    # ─── Delete Session (Soft + Hard) ──────────────────────────

    async def delete_session(
        self,
        db: AsyncSession,
        session_id: str,
        tenant_id: int,
        hard_delete: bool = False,
    ) -> bool:
        """
        Delete a chat session and its messages.
        Soft-delete: sets status to 'deleted', clears messages.
        Hard-delete: removes rows entirely.
        """
        # Verify session exists and belongs to tenant
        result = await db.execute(
            select(ChatSession).where(
                ChatSession.session_id == session_id,
                ChatSession.tenant_id == tenant_id,
            )
        )
        session = result.scalar_one_or_none()
        if not session:
            return False

        # Delete messages
        await db.execute(
            delete(ChatMessage).where(ChatMessage.session_id == session_id)
        )

        if hard_delete:
            await db.delete(session)
        else:
            # Soft-delete
            session.status = "deleted"
            session.message_count = 0
            session.last_message_at = None

        await db.flush()
        logger.info(f"ChatPersistence: {'hard' if hard_delete else 'soft'}-deleted session {session_id}")
        return True

    # ─── List Sessions (Admin/Analytics) ───────────────────────

    async def get_sessions_for_tenant(
        self,
        db: AsyncSession,
        tenant_id: int,
        status: str = "active",
        limit: int = 20,
    ) -> list[dict]:
        """List recent chat sessions for a tenant (admin dashboard)."""
        result = await db.execute(
            select(ChatSession)
            .where(
                ChatSession.tenant_id == tenant_id,
                ChatSession.status == status,
            )
            .order_by(ChatSession.last_message_at.desc().nullslast())
            .limit(limit)
        )
        sessions = result.scalars().all()

        return [
            {
                "session_id": s.session_id,
                "channel": s.channel,
                "language": s.language,
                "message_count": s.message_count,
                "last_message_at": s.last_message_at.isoformat() if s.last_message_at else None,
                "created_at": s.created_at.isoformat() if s.created_at else None,
                "metadata": s.metadata_ or {},
            }
            for s in sessions
        ]

    # ─── Internal Helpers ──────────────────────────────────────

    async def _get_or_create_session(
        self,
        db: AsyncSession,
        session_id: str,
        tenant_id: int,
        channel: str = "websocket",
        language: str | None = None,
    ) -> ChatSession:
        """Get existing session or create a new one."""
        result = await db.execute(
            select(ChatSession).where(ChatSession.session_id == session_id)
        )
        session = result.scalar_one_or_none()

        if session:
            return session

        session = ChatSession(
            session_id=session_id,
            tenant_id=tenant_id,
            channel=channel,
            language=language,
            status="active",
        )
        db.add(session)
        await db.flush()
        logger.info(f"ChatPersistence: created new session {session_id} for tenant {tenant_id}")
        return session


# ─── Singleton ──────────────────────────────────────────────
_service: ChatPersistenceService | None = None


def get_chat_persistence() -> ChatPersistenceService:
    global _service
    if _service is None:
        _service = ChatPersistenceService()
    return _service
