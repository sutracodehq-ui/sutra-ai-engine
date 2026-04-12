"""
User Feedback Collector — Ground-truth signal for AI quality.

Software Factory Principle: Quality Control via real user signals.

Collects thumbs up/down feedback on AI responses, stores in PostgreSQL,
and converts positive examples into training data for LoRA fine-tuning.

Architecture:
    User → 👍/👎 → FeedbackCollector → PostgreSQL
                                      → Positive examples → training/data/*.jsonl
                                      → PromptEngine (record win/loss)
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from sqlalchemy import Column, DateTime, Float, Integer, String, Text, Boolean, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.base import Base

logger = logging.getLogger(__name__)


# ─── SQLAlchemy Model ───────────────────────────────────────

class AgentFeedback(Base):
    """Stores user feedback on AI responses."""
    __tablename__ = "agent_feedback"

    id = Column(Integer, primary_key=True, autoincrement=True)
    agent_type = Column(String(100), nullable=False, index=True)
    prompt = Column(Text, nullable=False)
    response = Column(Text, nullable=False)
    is_positive = Column(Boolean, nullable=False)          # True = 👍, False = 👎
    quality_score = Column(Float, nullable=True)            # Optional 1-5 star rating
    feedback_text = Column(Text, nullable=True)             # Optional written feedback
    user_id = Column(String(100), nullable=True, index=True)
    brand_id = Column(String(100), nullable=True)
    system_prompt_version = Column(Integer, nullable=True)  # Links to AgentOptimization
    exported_to_training = Column(Boolean, default=False)   # Whether exported as JSONL
    created_at = Column(DateTime, default=func.now())


# ─── Feedback Collector Service ─────────────────────────────

class FeedbackCollector:
    """
    Collects, stores, and exports user feedback for AI training.

    Features:
    - Store thumbs up/down with optional comments
    - Export positive feedback as JSONL training data
    - Feed results into PromptEngine for A/B tracking
    - Per-agent feedback analytics
    """

    def __init__(self, db: AsyncSession):
        self._db = db
        self._training_dir = Path("training/data")
        try:
            self._training_dir.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            self._training_dir = None

    async def record(
        self,
        agent_type: str,
        prompt: str,
        response: str,
        is_positive: bool,
        quality_score: float | None = None,
        feedback_text: str | None = None,
        user_id: str | None = None,
        brand_id: str | None = None,
        system_prompt_version: int | None = None,
    ) -> AgentFeedback:
        """
        Record user feedback on an AI response.
        
        Positive feedback is automatically queued for training export.
        """
        feedback = AgentFeedback(
            agent_type=agent_type,
            prompt=prompt,
            response=response,
            is_positive=is_positive,
            quality_score=quality_score,
            feedback_text=feedback_text,
            user_id=user_id,
            brand_id=brand_id,
            system_prompt_version=system_prompt_version,
        )

        self._db.add(feedback)
        await self._db.commit()

        emoji = "👍" if is_positive else "👎"
        logger.info(f"Feedback: {emoji} for {agent_type} (score={quality_score})")

        return feedback

    async def export_positive_examples(self) -> dict:
        """
        Export all unexported positive feedback as JSONL training data.
        
        Only 👍 responses become training examples.
        This is the ground-truth signal that makes LoRA fine-tuning effective.
        """
        # Fetch unexported positive feedback
        stmt = (
            select(AgentFeedback)
            .where(
                AgentFeedback.is_positive == True,
                AgentFeedback.exported_to_training == False,
            )
            .order_by(AgentFeedback.created_at)
        )
        result = await self._db.execute(stmt)
        feedbacks = result.scalars().all()

        if not feedbacks:
            return {"exported": 0, "message": "No new positive feedback to export"}

        # Write as JSONL
        output_path = self._training_dir / f"user_feedback_{datetime.now().strftime('%Y%m%d')}.jsonl"
        exported = 0

        with open(output_path, "a") as f:
            for fb in feedbacks:
                example = {
                    "messages": [
                        {"role": "system", "content": f"You are the {fb.agent_type} agent."},
                        {"role": "user", "content": fb.prompt},
                        {"role": "assistant", "content": fb.response},
                    ],
                    "source": "user_feedback",
                    "quality_score": fb.quality_score,
                }
                f.write(json.dumps(example, ensure_ascii=False) + "\n")

                # Mark as exported
                fb.exported_to_training = True
                exported += 1

        await self._db.commit()

        logger.info(f"FeedbackCollector: exported {exported} positive examples → {output_path}")
        return {"exported": exported, "file": str(output_path)}

    async def get_agent_stats(self, agent_type: str) -> dict:
        """Get feedback stats for a specific agent."""
        # Total feedback
        total_stmt = select(func.count()).where(AgentFeedback.agent_type == agent_type)
        total = (await self._db.execute(total_stmt)).scalar() or 0

        # Positive
        pos_stmt = select(func.count()).where(
            AgentFeedback.agent_type == agent_type,
            AgentFeedback.is_positive == True,
        )
        positive = (await self._db.execute(pos_stmt)).scalar() or 0

        # Average score
        avg_stmt = select(func.avg(AgentFeedback.quality_score)).where(
            AgentFeedback.agent_type == agent_type,
            AgentFeedback.quality_score.isnot(None),
        )
        avg_score = (await self._db.execute(avg_stmt)).scalar()

        return {
            "agent_type": agent_type,
            "total_feedback": total,
            "positive": positive,
            "negative": total - positive,
            "satisfaction_rate": round(positive / max(total, 1) * 100, 1),
            "avg_quality_score": round(avg_score or 0, 2),
        }

    async def get_all_stats(self) -> list[dict]:
        """Get feedback stats for all agents."""
        stmt = select(AgentFeedback.agent_type).distinct()
        result = await self._db.execute(stmt)
        agent_types = [row[0] for row in result.all()]

        stats = []
        for agent_type in agent_types:
            stat = await self.get_agent_stats(agent_type)
            stats.append(stat)

        return sorted(stats, key=lambda x: x["total_feedback"], reverse=True)
