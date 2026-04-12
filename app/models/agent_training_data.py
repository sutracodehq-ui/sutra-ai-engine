"""Agent Training Data — self-learning records (OPRO, TextGrad, few-shot examples)."""

from sqlalchemy import BigInteger, Boolean, Float, ForeignKey, JSON, String
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TenantScopedMixin, TimestampMixin


class AgentTrainingData(Base, TenantScopedMixin, TimestampMixin):
    __tablename__ = "agent_training_data"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    tenant_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("tenants.id"), nullable=False, index=True)
    agent_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)

    # Source: meta_prompt | edit_diff | few_shot | ab_variant | user_upload
    source_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)

    # The training data payload
    data: Mapped[dict] = mapped_column(JSON, nullable=False)

    # Quality score for ranking
    score: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)

    # Active flag — only active records are injected into prompts
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    def __repr__(self) -> str:
        return f"<AgentTrainingData {self.id} type={self.source_type} agent={self.agent_type}>"
