"""API Key model — dedicated table for multi-key tenant auth with expiry + scopes."""

from datetime import datetime
from typing import Optional

from sqlalchemy import BigInteger, Boolean, DateTime, ForeignKey, JSON, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin


class ApiKey(Base, TimestampMixin):
    __tablename__ = "api_keys"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    tenant_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True)

    # ─── Key Data ───────────────────────────────────────
    key_hash: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    key_prefix: Mapped[str] = mapped_column(String(30), nullable=False)  # "sk_live_abc12345..."

    # ─── Classification ─────────────────────────────────
    environment: Mapped[str] = mapped_column(String(10), nullable=False, default="live")  # "live" | "test"
    label: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)  # "Frontend App", "CI/CD Pipeline"

    # ─── Scopes (future-ready, default = full access) ───
    # Default ["*"] grants all permissions. Future: ["agents:read", "voice:*"]
    scopes: Mapped[list | None] = mapped_column(JSON, default=lambda: ["*"])

    # ─── Lifecycle ──────────────────────────────────────
    expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)  # null = never
    last_used_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    # ─── Relationships ──────────────────────────────────
    tenant = relationship("Tenant", back_populates="api_keys")

    @property
    def is_expired(self) -> bool:
        """Check if key has passed its expiry."""
        if self.expires_at is None:
            return False
        from datetime import timezone
        return datetime.now(timezone.utc) > self.expires_at

    def __repr__(self) -> str:
        return f"<ApiKey {self.key_prefix} env={self.environment} tenant={self.tenant_id}>"
