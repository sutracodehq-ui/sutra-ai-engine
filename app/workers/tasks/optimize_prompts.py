"""
Optimize Prompts — Celery periodic task.

Self-Learning Tier 3: Runs daily to analyze agent failures
and generate improved system prompts via Groq meta-LLM.

Schedule: Every day at 3:00 AM (configurable via Celery Beat).
"""

import logging

from app.workers.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, name="optimize_prompts", max_retries=1)
def optimize_prompts(self):
    """
    Daily prompt optimization cycle.

    Finds agents with 3+ failures, generates improved prompts via Groq,
    and stores them as A/B test candidates.
    """
    import asyncio
    asyncio.run(_do_optimize())


async def _do_optimize():
    """Async implementation of the prompt optimization via Brain."""
    from app.db.session import async_session_factory
    from app.services.intelligence.brain import get_brain

    async with async_session_factory() as db:
        brain = get_brain()
        result = await brain.run_optimization_cycle(db)

    if result["optimized"]:
        logger.info(
            f"Prompt optimization complete: optimized={result['optimized']}, "
            f"skipped={result['skipped']}, errors={len(result['errors'])}"
        )
    else:
        logger.info("No agents needed optimization this cycle.")
