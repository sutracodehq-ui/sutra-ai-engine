"""UrlAnalysis model — stores scraped website digital footprints for training."""

from sqlalchemy import BigInteger, JSON, String, Text, Integer, Float, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TenantScopedMixin, TimestampMixin


class UrlAnalysis(Base, TenantScopedMixin, TimestampMixin):
    __tablename__ = "url_analyses"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    tenant_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("tenants.id"), nullable=False, index=True)
    
    # Target
    url: Mapped[str] = mapped_column(String(512), nullable=False, index=True)
    domain: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    
    # Scores
    seo_score: Mapped[int] = mapped_column(Integer, default=0)
    security_score: Mapped[int] = mapped_column(Integer, default=0)
    
    # Full scraped data (JSON blob for training)
    scraped_data: Mapped[dict | None] = mapped_column(JSON, default=dict)
    
    # Google indexing snapshot
    indexed_pages: Mapped[int] = mapped_column(Integer, default=0)
    
    # Tech stack snapshot
    tech_stack: Mapped[list | None] = mapped_column(JSON, default=list)
    
    # AI-generated report (saved for fine-tuning)
    ai_report: Mapped[str | None] = mapped_column(Text, nullable=True)
    
    # Relationship
    tenant = relationship("Tenant", backref="url_analyses")

    def __repr__(self) -> str:
        return f"<UrlAnalysis domain={self.domain} seo={self.seo_score}>"
