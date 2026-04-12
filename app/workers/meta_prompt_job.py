"""
Meta-Prompt Job — Celery task to periodically optimize agent prompts.
"""

import logging
from typing import List

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.workers.celery_app import celery_app
from app.dependencies import get_db_context
from app.models.agent_feedback import AgentFeedback
from app.models.agent_optimization import AgentOptimization
from app.services.learning.meta_prompt import MetaPromptService
from app.config import get_settings

logger = logging.getLogger(__name__)


@celery_app.task(name="app.workers.meta_prompt_job.meta_prompt_optimize")
async def meta_prompt_optimize():
    """
    Periodic job that finds agents with high feedback counts and optimizes them.
    """
    settings = get_settings()
    if not settings.ai_meta_prompt_enabled:
        return

    async with get_db_context() as db:
        # 1. Find agents that cross the threshold
        agents_to_optimize = await _get_agents_needing_optimization(db, settings.ai_meta_prompt_threshold)
        
        for agent_type in agents_to_optimize:
            logger.info(f"Learn Loop: Optimizing agent '{agent_type}'...")
            
            # 2. Generate new prompt via OPRO
            new_prompt = await MetaPromptService.optimize_agent(db, agent_type)
            if not new_prompt:
                continue

            # 3. Store the optimization (inactive by default for safety / review)
            # Find current max version
            version_stmt = select(func.max(AgentOptimization.version)).where(AgentOptimization.agent_type == agent_type)
            current_max = (await db.execute(version_stmt)).scalar() or 0
            
            optimization = AgentOptimization(
                agent_type=agent_type,
                version=current_max + 1,
                prompt_text=new_prompt,
                notes=f"Evolved via OPRO using {settings.ai_meta_prompt_model}",
                is_active=False  # Requires human review or A/B test trigger
            )
            db.add(optimization)
            await db.commit()
            
            logger.info(f"Learn Loop: Created version {optimization.version} for '{agent_type}'")


async def _get_agents_needing_optimization(db: AsyncSession, threshold: int) -> List[str]:
    """Get list of agent types that have enough feedback for a meaningful learning step."""
    # This is a simplified check: agents with > threshold feedback total
    # In a real system, we'd check for feedback *since the last optimization*.
    stmt = (
        select(AgentFeedback.agent_type)
        .group_by(AgentFeedback.agent_type)
        .having(func.count(AgentFeedback.id) >= threshold)
    )
    result = await db.execute(stmt)
    return list(result.scalars().all())
