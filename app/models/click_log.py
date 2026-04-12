"""ClickLog model — stores historical click data and fraud scores."""

from sqlalchemy import BigInteger, Boolean, JSON, String, Text, Float, Integer, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TenantScopedMixin, TimestampMixin


class ClickLog(Base, TenantScopedMixin, TimestampMixin):
    __tablename__ = "click_logs"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    tenant_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("tenants.id"), nullable=False, index=True)
    ad_id: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    
    ip_address: Mapped[str | None] = mapped_column(String(45), nullable=True, index=True)
    user_agent: Mapped[str | None] = mapped_column(Text, nullable=True)
    referrer: Mapped[str | None] = mapped_column(String(512), nullable=True)
    
    # Fraud Scoring
    fraud_score: Mapped[int] = mapped_column(Integer, default=0)
    is_blocked: Mapped[bool] = mapped_column(Boolean, default=False)
    action: Mapped[str] = mapped_column(String(20), default="ALLOW")
    reasons: Mapped[list | None] = mapped_column(JSON, default=list) # List of dictionaries
    
    # Behavioral Data (Snapshot)
    client_data: Mapped[dict | None] = mapped_column(JSON, default=dict)
    
    # Relationship
    tenant = relationship("Tenant", back_populates="click_logs")

    def __repr__(self) -> str:
        return f"<ClickLog ad={self.ad_id} score={self.fraud_score} blocked={self.is_blocked}>"
