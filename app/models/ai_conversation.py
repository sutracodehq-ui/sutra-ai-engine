"""AI Conversation model — groups related AI tasks into a thread."""

from sqlalchemy import BigInteger, ForeignKey, JSON, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TenantScopedMixin, TimestampMixin


class AiConversation(Base, TenantScopedMixin, TimestampMixin):
    __tablename__ = "ai_conversations"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    tenant_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("tenants.id"), nullable=False, index=True)
    agent_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    external_user_id: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    metadata_: Mapped[dict | None] = mapped_column("metadata", JSON, default=dict)

    # Relationships
    tenant = relationship("Tenant", back_populates="conversations")
    tasks = relationship("AiTask", back_populates="conversation", order_by="AiTask.created_at", lazy="selectin")

    def __repr__(self) -> str:
        return f"<AiConversation {self.id} agent={self.agent_type}>"
