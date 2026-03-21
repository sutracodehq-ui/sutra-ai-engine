"""Agent Optimization model — tracks evolved prompt versions with performance metrics."""

from sqlalchemy import BigInteger, Boolean, Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin


class AgentOptimization(Base, TimestampMixin):
    """
    Stores evolved versions of agent prompts with live performance tracking.
    
    Self-Optimizing Prompt Engine:
    - Prompts are living, versioned assets in the database
    - Each version tracks trial_count and total_score
    - The engine auto-promotes candidates that outperform the champion
    - Status lifecycle: candidate → active → champion → retired
    """
    __tablename__ = "agent_optimizations"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    agent_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    
    # The actual optimized prompt text
    prompt_text: Mapped[str] = mapped_column(Text, nullable=False)
    
    # Metadata
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    
    # Status: candidate | active | champion | retired
    status: Mapped[str] = mapped_column(String(20), default="candidate", nullable=False)
    
    # ─── Performance Tracking ──────────────────────────
    trial_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    total_score: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    win_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    loss_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    @property
    def avg_score(self) -> float:
        """Running average quality score."""
        if self.trial_count == 0:
            return 0.0
        return round(self.total_score / self.trial_count, 2)

    @property
    def win_rate(self) -> float:
        """Percentage of trials where quality gate passed."""
        total = self.win_count + self.loss_count
        if total == 0:
            return 0.0
        return round(self.win_count / total * 100, 1)

    def record_trial(self, score: float, passed: bool) -> None:
        """Record a trial result (called by the prompt engine after each use)."""
        self.trial_count += 1
        self.total_score += score
        if passed:
            self.win_count += 1
        else:
            self.loss_count += 1

    def __repr__(self) -> str:
        return (
            f"<AgentOptimization {self.agent_type} v{self.version} "
            f"status={self.status} avg={self.avg_score} trials={self.trial_count}>"
        )

