"""Tenant model — each consuming product (Tryambaka, e-commerce, etc.)."""

from sqlalchemy import BigInteger, Boolean, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin


class Tenant(Base, TimestampMixin):
    __tablename__ = "tenants"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    slug: Mapped[str] = mapped_column(String(100), unique=True, nullable=False, index=True)
    api_key_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    api_key_prefix: Mapped[str] = mapped_column(String(20), nullable=False)  # e.g. "sk_live_abc..."
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    # Tenant-level config overrides
    config: Mapped[dict | None] = mapped_column(JSON, default=dict)

    # Rate limits override (null = use global defaults)
    rate_limits: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    # Contact / metadata
    contact_email: Mapped[str | None] = mapped_column(String(255), nullable=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Relationships
    conversations = relationship("AiConversation", back_populates="tenant", lazy="dynamic")
    tasks = relationship("AiTask", back_populates="tenant", lazy="dynamic")
    voice_profiles = relationship("VoiceProfile", back_populates="tenant", lazy="dynamic")

    def __repr__(self) -> str:
        return f"<Tenant {self.slug}>"
