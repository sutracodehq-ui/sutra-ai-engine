"""
Evolution Service — calculates winner between Active and Candidate prompts.

Logic:
1. Fetch feedback scores for Version A (Active) and Version B (Candidate).
2. If B has > N samples and average score is > A by Threshold → B becomes Active.
3. Old A is marked as historical (inactive).
"""

import logging
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.agent_optimization import AgentOptimization
from app.models.ai_task import AiTask
from app.models.agent_feedback import AgentFeedback

logger = logging.getLogger(__name__)


class EvolutionService:
    """Service to handle prompt auto-promotion."""

    @classmethod
    async def run_evolution_cycle(cls, db: AsyncSession, agent_type: str):
        """Analyze current A/B test and promote if candidate is better."""
        
        # 1. Fetch current Active and Candidate
        stmt = (
            select(AgentOptimization)
            .where(AgentOptimization.agent_type == agent_type)
            .order_by(AgentOptimization.is_active.desc(), AgentOptimization.version.desc())
            .limit(10)
        )
        result = await db.execute(stmt)
        opts = result.scalars().all()
        
        active = next((o for o in opts if o.is_active), None)
        candidate = next((o for o in opts if not o.is_active), None)

        if not active or not candidate:
            logger.info(f"Evolution: No active/candidate pair for {agent_type}. Skipping.")
            return

        # 2. Calculate scores
        active_score = await cls._get_average_score(db, active.id)
        candidate_score = await cls._get_average_score(db, candidate.id)
        
        logger.info(f"Evolution [{agent_type}]: Active v{active.version}({active_score['avg']}) vs Candidate v{candidate.version}({candidate_score['avg']})")

        # 3. Promotion Logic
        # Min samples (10) and significant improvement (e.g. 0.1 delta)
        if candidate_score["count"] >= 10 and candidate_score["avg"] > (active_score["avg"] + 0.1):
            logger.info(f"🏆 Evolution: Promoting Candidate v{candidate.version} to ACTIVE for {agent_type}")
            
            # Deactivate old
            active.is_active = False
            # Activate new
            candidate.is_active = True
            
            await db.commit()
            return True

        return False

    @classmethod
    async def _get_average_score(cls, db: AsyncSession, opt_id: int) -> dict:
        """Calculate average score and sample count for an optimization ID."""
        # Join AiTask -> AgentFeedback
        stmt = (
            select(func.avg(AgentFeedback.score), func.count(AgentFeedback.id))
            .join(AiTask, AiTask.id == AgentFeedback.task_id)
            .where(AiTask.agent_optimization_id == opt_id)
        )
        result = await db.execute(stmt)
        avg, count = result.fetchone()
        
        return {
            "avg": float(avg) if avg else 0.0,
            "count": int(count) if count else 0
        }
