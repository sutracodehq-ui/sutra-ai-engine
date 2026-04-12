"""Voice Profile — tenant-specific tone/style for AI outputs."""

from sqlalchemy import BigInteger, Boolean, ForeignKey, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TenantScopedMixin, TimestampMixin


class VoiceProfile(Base, TenantScopedMixin, TimestampMixin):
    __tablename__ = "voice_profiles"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    tenant_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("tenants.id"), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    is_default: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    # Voice attributes: tone, formality, personality, etc.
    tone_attributes: Mapped[dict | None] = mapped_column(JSON, default=dict)

    # Direct system prompt modifier injected into LLM calls
    system_modifier: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Relationships
    tenant = relationship("Tenant", back_populates="voice_profiles")

    def to_system_prompt_modifier(self) -> str:
        """Build the voice instruction block for the system prompt."""
        parts = []
        if self.system_modifier:
            parts.append(self.system_modifier)
        if self.tone_attributes:
            attrs = self.tone_attributes
            if "tone" in attrs:
                parts.append(f"Tone: {attrs['tone']}")
            if "formality" in attrs:
                parts.append(f"Formality: {attrs['formality']}")
            if "personality" in attrs:
                parts.append(f"Personality: {attrs['personality']}")
            if "vocabulary" in attrs:
                parts.append(f"Vocabulary: {attrs['vocabulary']}")
        return "\n".join(parts) if parts else ""

    def __repr__(self) -> str:
        return f"<VoiceProfile {self.name} tenant={self.tenant_id}>"
