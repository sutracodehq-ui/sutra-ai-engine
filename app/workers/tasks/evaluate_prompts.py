"""
Evaluate Prompt Promotions — Celery periodic task.

Self-Optimizing Prompt Engine: Runs daily to check if any candidate
prompts have earned promotion to champion status based on their
trial results vs. the current champion.
"""

import logging

from app.workers.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, name="evaluate_prompt_promotions", max_retries=1)
def evaluate_prompt_promotions(self):
    """
    Daily promotion evaluation cycle.

    Checks all candidates with enough trials and promotes any that
    beat the current champion by the required margin.
    """
    import asyncio
    asyncio.run(_do_evaluate())


async def _do_evaluate():
    """Async implementation of the promotion evaluation."""
    from app.db.session import async_session_factory
    from app.services.intelligence.prompt_engine import PromptEngine

    async with async_session_factory() as db:
        engine = PromptEngine(db)
        result = await engine.evaluate_all_promotions()

    if result["promoted"]:
        logger.info(f"Prompt promotions: promoted={result['promoted']}")
    if result["retired"]:
        logger.info(f"Prompt retirements: retired={result['retired']}")
    if result["pending"]:
        logger.info(f"Pending candidates: {result['pending']}")
    if not any(result.values()):
        logger.info("No prompt promotions needed this cycle.")
