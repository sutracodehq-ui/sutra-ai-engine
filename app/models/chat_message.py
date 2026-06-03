"""ChatMessage model — individual messages in a chat session.

Each row = one message (user or assistant).
Linked to ChatSession by session_id string (not FK to id).
"""

from sqlalchemy import BigInteger, Float, ForeignKey, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin


class ChatMessage(Base, TimestampMixin):
    __tablename__ = "chat_messages"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)

    # Links to chat_sessions.session_id (the browser UUID)
    session_id: Mapped[str] = mapped_column(
        String(100),
        ForeignKey("chat_sessions.session_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Multi-tenancy (denormalized for fast tenant-scoped queries)
    tenant_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("tenants.id"), nullable=False, index=True
    )

    # Message content
    role: Mapped[str] = mapped_column(String(20), nullable=False)  # user / assistant / system
    content: Mapped[str] = mapped_column(Text, nullable=False)

    # Interactive actions attached to this message (JSON array)
    actions: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    # AI confidence score (null for user messages)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Extensible metadata (tokens_used, driver, model, etc.)
    metadata_: Mapped[dict | None] = mapped_column("metadata", JSON, default=dict)

    # Relationships
    session = relationship("ChatSession", back_populates="messages")

    def __repr__(self) -> str:
        preview = self.content[:40] if self.content else ""
        return f"<ChatMessage {self.id} role={self.role} '{preview}...'>"
