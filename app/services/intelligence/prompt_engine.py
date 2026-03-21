"""
Self-Optimizing Prompt Engine — Autonomous prompt lifecycle manager.

The OPRO (Optimization by PROmpting) engine that closes the loop:

    ┌─────────────────────────────────────────────────────┐
    │                                                     │
    │   YAML Config ──→ Candidate ──→ Active ──→ Champion │
    │       ↑                                     │       │
    │       │              ← retirement ←─────────┘       │
    │       │                                             │
    │   MetaPromptOptimizer                               │
    │   (generates new candidates from failure analysis)  │
    │                                                     │
    └─────────────────────────────────────────────────────┘

Lifecycle:
1. YAML → baseline prompt (version 0, status="champion")
2. MetaPromptOptimizer → generates candidate (status="candidate")
3. A/B testing routes 10-20% traffic to candidate
4. PromptEngine tracks quality scores per version
5. After N trials: if candidate.avg_score > champion.avg_score → PROMOTE
6. Old champion → status="retired"
7. Repeat forever
"""

import logging
from typing import Optional, Tuple

from sqlalchemy import select, and_, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.models.agent_optimization import AgentOptimization

logger = logging.getLogger(__name__)

# Minimum trials before a candidate can be promoted
MIN_TRIALS_FOR_PROMOTION = 10
# Minimum score advantage for promotion (candidate must beat champion by this margin)
PROMOTION_MARGIN = 0.5
# Minimum win rate for promotion
MIN_WIN_RATE = 60.0


class PromptEngine:
    """
    Self-optimizing prompt manager.

    Responsibilities:
    1. select_prompt() — pick which prompt version to use (champion or candidate)
    2. record_result() — track quality score for the version used
    3. evaluate_promotions() — check if any candidate should be promoted
    4. bootstrap() — seed the DB with YAML baseline as initial champion
    """

    def __init__(self, db: AsyncSession):
        self._db = db

    # ─── Selection ──────────────────────────────────────────

    async def select_prompt(self, agent_type: str) -> Tuple[str, Optional[int]]:
        """
        Select the prompt to use for an agent.

        Strategy:
        - 80% traffic: use the champion prompt
        - 20% traffic: use a candidate (if available) for A/B testing
        - Fallback: YAML config
        """
        import random

        settings = get_settings()
        explore_rate = settings.ai_explore_rate

        # Try candidate with explore_rate probability
        if random.random() < explore_rate:
            candidate = await self._get_best_candidate(agent_type)
            if candidate:
                logger.info(
                    f"PromptEngine: 🧪 Testing candidate v{candidate.version} for {agent_type} "
                    f"(trials={candidate.trial_count}, avg={candidate.avg_score})"
                )
                return candidate.prompt_text, candidate.id

        # Use champion
        champion = await self._get_champion(agent_type)
        if champion:
            return champion.prompt_text, champion.id

        # Use active (legacy field support)
        active = await self._get_active(agent_type)
        if active:
            return active.prompt_text, active.id

        # No DB prompts — fall back to YAML (handled by BaseAgent)
        return None, None

    async def _get_champion(self, agent_type: str) -> Optional[AgentOptimization]:
        """Get the current champion prompt."""
        stmt = (
            select(AgentOptimization)
            .where(and_(
                AgentOptimization.agent_type == agent_type,
                AgentOptimization.status == "champion",
            ))
            .order_by(AgentOptimization.version.desc())
            .limit(1)
        )
        result = await self._db.execute(stmt)
        return result.scalar_one_or_none()

    async def _get_best_candidate(self, agent_type: str) -> Optional[AgentOptimization]:
        """Get the best-performing candidate."""
        stmt = (
            select(AgentOptimization)
            .where(and_(
                AgentOptimization.agent_type == agent_type,
                AgentOptimization.status == "candidate",
            ))
            .order_by(AgentOptimization.version.desc())
            .limit(1)
        )
        result = await self._db.execute(stmt)
        return result.scalar_one_or_none()

    async def _get_active(self, agent_type: str) -> Optional[AgentOptimization]:
        """Legacy: get is_active=True prompt."""
        stmt = (
            select(AgentOptimization)
            .where(and_(
                AgentOptimization.agent_type == agent_type,
                AgentOptimization.is_active == True,
            ))
            .order_by(AgentOptimization.version.desc())
            .limit(1)
        )
        result = await self._db.execute(stmt)
        return result.scalar_one_or_none()

    # ─── Recording ──────────────────────────────────────────

    async def record_result(self, optimization_id: int, quality_score: float, passed: bool) -> None:
        """
        Record a quality score for a prompt version.
        Called after every agent execution with the quality gate result.
        """
        try:
            stmt = select(AgentOptimization).where(AgentOptimization.id == optimization_id)
            result = await self._db.execute(stmt)
            opt = result.scalar_one_or_none()

            if not opt:
                return

            opt.record_trial(quality_score, passed)
            await self._db.commit()

            logger.debug(
                f"PromptEngine: recorded trial for {opt.agent_type} v{opt.version} "
                f"(score={quality_score}, avg={opt.avg_score}, trials={opt.trial_count})"
            )

            # Check for auto-promotion after recording
            if opt.status == "candidate" and opt.trial_count >= MIN_TRIALS_FOR_PROMOTION:
                await self._check_promotion(opt)

        except Exception as e:
            logger.warning(f"PromptEngine: record_result failed: {e}")

    # ─── Promotion Engine ───────────────────────────────────

    async def _check_promotion(self, candidate: AgentOptimization) -> None:
        """
        Check if a candidate should be promoted to champion.

        Criteria:
        1. Candidate has >= MIN_TRIALS_FOR_PROMOTION trials
        2. Candidate avg_score > champion avg_score + PROMOTION_MARGIN
        3. Candidate win_rate >= MIN_WIN_RATE
        """
        champion = await self._get_champion(candidate.agent_type)

        # No champion exists — auto-promote
        if not champion:
            await self._promote(candidate)
            return

        # Check promotion criteria
        score_beats = candidate.avg_score > (champion.avg_score + PROMOTION_MARGIN)
        win_rate_ok = candidate.win_rate >= MIN_WIN_RATE

        if score_beats and win_rate_ok:
            logger.info(
                f"PromptEngine: 🏆 PROMOTING {candidate.agent_type} "
                f"v{candidate.version} (avg={candidate.avg_score}, win={candidate.win_rate}%) "
                f"over champion v{champion.version} (avg={champion.avg_score}, win={champion.win_rate}%)"
            )
            champion.status = "retired"
            champion.is_active = False
            await self._promote(candidate)
        else:
            # Check if candidate is so bad it should be retired
            if candidate.trial_count >= MIN_TRIALS_FOR_PROMOTION * 2 and candidate.win_rate < 30.0:
                logger.info(
                    f"PromptEngine: ❌ RETIRING weak candidate {candidate.agent_type} "
                    f"v{candidate.version} (avg={candidate.avg_score}, win={candidate.win_rate}%)"
                )
                candidate.status = "retired"
                await self._db.commit()

    async def _promote(self, opt: AgentOptimization) -> None:
        """Promote a prompt version to champion."""
        opt.status = "champion"
        opt.is_active = True
        await self._db.commit()
        logger.info(f"PromptEngine: ✅ v{opt.version} is now CHAMPION for {opt.agent_type}")

    # ─── Bulk Operations ────────────────────────────────────

    async def evaluate_all_promotions(self) -> dict:
        """
        Evaluate all candidates across all agents for promotion.
        Called by the daily Celery task.
        """
        result = {"promoted": [], "retired": [], "pending": []}

        # Get all candidates with enough trials
        stmt = (
            select(AgentOptimization)
            .where(and_(
                AgentOptimization.status == "candidate",
                AgentOptimization.trial_count >= MIN_TRIALS_FOR_PROMOTION,
            ))
        )
        candidates_result = await self._db.execute(stmt)
        candidates = candidates_result.scalars().all()

        for candidate in candidates:
            old_status = candidate.status
            await self._check_promotion(candidate)
            await self._db.refresh(candidate)

            if candidate.status == "champion":
                result["promoted"].append(f"{candidate.agent_type} v{candidate.version}")
            elif candidate.status == "retired":
                result["retired"].append(f"{candidate.agent_type} v{candidate.version}")
            else:
                result["pending"].append(
                    f"{candidate.agent_type} v{candidate.version} "
                    f"(avg={candidate.avg_score}, win={candidate.win_rate}%, trials={candidate.trial_count})"
                )

        return result

    async def bootstrap_from_yaml(self, agent_type: str, yaml_prompt: str) -> AgentOptimization:
        """
        Seed the DB with the YAML baseline as the initial champion.
        Only creates if no champion exists for this agent.
        """
        existing = await self._get_champion(agent_type)
        if existing:
            return existing

        opt = AgentOptimization(
            agent_type=agent_type,
            version=0,
            prompt_text=yaml_prompt,
            notes="Bootstrapped from YAML config",
            is_active=True,
            status="champion",
        )
        self._db.add(opt)
        await self._db.commit()
        await self._db.refresh(opt)
        logger.info(f"PromptEngine: bootstrapped {agent_type} v0 as champion from YAML")
        return opt

    async def get_leaderboard(self, agent_type: str) -> list[dict]:
        """Get all prompt versions for an agent, ranked by avg_score."""
        stmt = (
            select(AgentOptimization)
            .where(AgentOptimization.agent_type == agent_type)
            .order_by(AgentOptimization.avg_score.desc() if hasattr(AgentOptimization, 'avg_score') 
                      else AgentOptimization.total_score.desc())
        )
        result = await self._db.execute(stmt)
        return [
            {
                "version": opt.version,
                "status": opt.status,
                "avg_score": opt.avg_score,
                "win_rate": opt.win_rate,
                "trial_count": opt.trial_count,
                "notes": opt.notes,
            }
            for opt in result.scalars().all()
        ]
