"""Agent Feedback model — user signals on AI output quality."""

from sqlalchemy import BigInteger, Float, ForeignKey, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TenantScopedMixin, TimestampMixin


class AgentFeedback(Base, TenantScopedMixin, TimestampMixin):
    __tablename__ = "agent_feedback"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    ai_task_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("ai_tasks.id"), nullable=False, index=True)
    tenant_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("tenants.id"), nullable=False, index=True)
    agent_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)

    # Feedback signal: accepted | edited | rejected | regenerated
    signal: Mapped[str] = mapped_column(String(20), nullable=False)

    # What the user changed (field → {original, edited})
    user_edit: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    # Computed quality score from signal
    quality_score: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)

    # Optional user comment
    comment: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Relationships
    task = relationship("AiTask", back_populates="feedback")

    SIGNAL_SCORES = {
        "accepted": 1.0,
        "edited": 0.5,
        "rejected": -1.0,
        "regenerated": -0.5,
    }

    @classmethod
    def score_for_signal(cls, signal: str) -> float:
        return cls.SIGNAL_SCORES.get(signal, 0.0)

    def __repr__(self) -> str:
        return f"<AgentFeedback {self.id} signal={self.signal}>"
