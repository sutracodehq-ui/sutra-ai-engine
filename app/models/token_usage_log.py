"""Token Usage Log — per-tenant, per-model cost tracking."""

from datetime import date

from sqlalchemy import BigInteger, Date, Float, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin


class TokenUsageLog(Base, TimestampMixin):
    __tablename__ = "token_usage_logs"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    tenant_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("tenants.id"), nullable=False, index=True)
    driver: Mapped[str] = mapped_column(String(50), nullable=False)
    model: Mapped[str] = mapped_column(String(100), nullable=False)
    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    completion_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    log_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)

    def __repr__(self) -> str:
        return f"<TokenUsageLog {self.driver}/{self.model} tenant={self.tenant_id}>"
