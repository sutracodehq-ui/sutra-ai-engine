"""ClickFeedback model — stores human labels for click fraud verification."""

from sqlalchemy import BigInteger, Boolean, ForeignKey, String, Text, Integer
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin


class ClickFeedback(Base, TimestampMixin):
    __tablename__ = "click_feedback"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    click_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("click_logs.id"), nullable=False, index=True)
    
    # Human label: is_fraud (True = confirmed bot, False = confirmed human)
    is_fraud: Mapped[bool] = mapped_column(Boolean, nullable=False)
    
    # Optional feedback details
    reason: Mapped[str | None] = mapped_column(String(100), nullable=True)
    comment: Mapped[str | None] = mapped_column(Text, nullable=True)
    
    # Relationship
    click_log = relationship("ClickLog", backref="feedback")

    def __repr__(self) -> str:
        return f"<ClickFeedback click={self.click_id} is_fraud={self.is_fraud}>"
