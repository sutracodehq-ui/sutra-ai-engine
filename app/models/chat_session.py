"""ChatSession model — persistent chatbot conversation sessions.

Software Factory: Futuristic schema with fields for cross-device sync,
omnichannel support, lead scoring, and analytics — all ready from day 1.
"""

from datetime import datetime

from sqlalchemy import BigInteger, DateTime, ForeignKey, Integer, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin


class ChatSession(Base, TimestampMixin):
    __tablename__ = "chat_sessions"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)

    # Browser-generated UUID — persisted in localStorage
    session_id: Mapped[str] = mapped_column(String(100), unique=True, nullable=False, index=True)

    # Multi-tenancy
    tenant_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("tenants.id"), nullable=False, index=True
    )

    # Visitor identity (futuristic: cross-device linking)
    visitor_fingerprint: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)

    # Channel tracking (omnichannel ready)
    channel: Mapped[str] = mapped_column(String(20), default="websocket", nullable=False)

    # Language detection
    language: Mapped[str | None] = mapped_column(String(10), nullable=True)

    # Lifecycle: active → archived → deleted
    status: Mapped[str] = mapped_column(String(20), default="active", nullable=False, index=True)

    # Extensible metadata (tags, lead_score, sentiment, etc.)
    metadata_: Mapped[dict | None] = mapped_column("metadata", JSON, default=dict)

    # Denormalized counters for fast queries
    message_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    last_message_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # Relationships
    tenant = relationship("Tenant", backref="chat_sessions")
    messages = relationship(
        "ChatMessage",
        back_populates="session",
        order_by="ChatMessage.created_at",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    def __repr__(self) -> str:
        return f"<ChatSession {self.session_id} tenant={self.tenant_id} msgs={self.message_count}>"
