"""Tenant model — each consuming product (Tryambaka, e-commerce, etc.)."""

from sqlalchemy import BigInteger, Boolean, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin


class Tenant(Base, TimestampMixin):
    __tablename__ = "tenants"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    slug: Mapped[str] = mapped_column(String(100), unique=True, nullable=False, index=True)

    # ─── Dual API Keys (sandbox + production) ────────────
    # Production key (sk_live_*)
    live_key_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    live_key_prefix: Mapped[str] = mapped_column(String(30), nullable=False)
    # Sandbox key (sk_test_*) — same tenant, isolated data in future
    test_key_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    test_key_prefix: Mapped[str] = mapped_column(String(30), nullable=False)

    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    # Tenant-level config overrides
    config: Mapped[dict | None] = mapped_column(JSON, default=dict)

    # Rate limits override (null = use global defaults)
    rate_limits: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    # Contact / metadata
    contact_email: Mapped[str | None] = mapped_column(String(255), nullable=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    webhook_url: Mapped[str | None] = mapped_column(String(512), nullable=True)
    identity_org_id: Mapped[str | None] = mapped_column(String(100), nullable=True, index=True, unique=True)

    # Relationships
    api_keys = relationship("ApiKey", back_populates="tenant", lazy="dynamic", cascade="all, delete-orphan")
    conversations = relationship("AiConversation", back_populates="tenant", lazy="dynamic")
    tasks = relationship("AiTask", back_populates="tenant", lazy="dynamic")
    voice_profiles = relationship("VoiceProfile", back_populates="tenant", lazy="dynamic")
    click_logs = relationship("ClickLog", back_populates="tenant", lazy="dynamic")

    def __repr__(self) -> str:
        return f"<Tenant {self.slug}>"
