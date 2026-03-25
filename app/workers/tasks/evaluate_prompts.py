"""
Evaluate Prompts — Legacy Stub.

Formerly ran daily to promote candidate prompts to champions.
Now handled automatically by Brain.record_result().
"""

import logging
from app.workers.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, name="evaluate_prompts", max_retries=1)
def evaluate_prompts(self):
    """Legacy evaluation worker redirect."""
    import asyncio
    asyncio.run(_do_evaluate())


async def _do_evaluate():
    """Legacy evaluation redirected to Brain."""
    logger.info("Evaluation task completed via Brain flow (auto-promotion active).")
