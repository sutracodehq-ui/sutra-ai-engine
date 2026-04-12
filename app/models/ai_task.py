"""AI Task model — individual LLM execution unit."""

from sqlalchemy import BigInteger, ForeignKey, Integer, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TenantScopedMixin, TimestampMixin


class AiTask(Base, TenantScopedMixin, TimestampMixin):
    __tablename__ = "ai_tasks"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    tenant_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("tenants.id"), nullable=False, index=True)
    conversation_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("ai_conversations.id"), nullable=True, index=True
    )
    external_user_id: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    agent_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)

    # Execution
    status: Mapped[str] = mapped_column(String(20), default="pending", nullable=False, index=True)
    prompt: Mapped[str] = mapped_column(Text, nullable=False)
    result: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    options: Mapped[dict | None] = mapped_column(JSON, default=dict)

    # Token accounting
    tokens_used: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    driver_used: Mapped[str | None] = mapped_column(String(50), nullable=True)
    model_used: Mapped[str | None] = mapped_column(String(100), nullable=True)

    # Error tracking
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Performance Attribution (for A/B testing)
    agent_optimization_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("agent_optimizations.id"), nullable=True
    )
    
    # Relationships
    tenant = relationship("Tenant", back_populates="tasks")
    conversation = relationship("AiConversation", back_populates="tasks")
    feedback = relationship("AgentFeedback", back_populates="task", uselist=False)

    def __repr__(self) -> str:
        return f"<AiTask {self.id} agent={self.agent_type} status={self.status}>"
