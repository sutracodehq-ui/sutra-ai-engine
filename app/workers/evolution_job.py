"""
Evolution Job — scheduled worker to promote winning prompts.
"""

import asyncio
import logging
from app.workers.celery_app import celery_app
from app.db.session import async_session_factory
from app.services.learning.prompt_evolution import EvolutionService

logger = logging.getLogger(__name__)


@celery_app.task(name="app.workers.evolution_job.run_prompt_evolution")
def run_prompt_evolution():
    """
    Background job to evaluate A/B test results and promote winners.
    Scheduled daily or weekly.
    """
    async def _run():
        async with async_session_factory() as db:
            # We iterate over common agent types to check for evolution
            # In a real system, we'd fetch all unique agent_types from AgentOptimization
            agent_types = ["marketing_agent", "support_agent", "creative_agent"]
            
            for agent_type in agent_types:
                try:
                    promoted = await EvolutionService.run_evolution_cycle(db, agent_type)
                    if promoted:
                        logger.info(f"🚀 Evolution: Agent '{agent_type}' has a new active prompt!")
                except Exception as e:
                    logger.error(f"Evolution failed for {agent_type}: {e}")

    # Run the async logic in the sync Celery worker context
    loop = asyncio.get_event_loop()
    if loop.is_running():
        # This shouldn't happen in a standard Celery worker, but safety first
        asyncio.ensure_future(_run())
    else:
        loop.run_until_complete(_run())
