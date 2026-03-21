"""
AI Evolution — Daily Celery background task.

Runs the Self-Evolution Engine cycle:
1. Discover new AI models
2. Benchmark against current model
3. Auto-upgrade if better found
4. Generate self-teaching training data
"""

import logging

from app.workers.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, name="evolve_ai", max_retries=1)
def evolve_ai(self):
    """
    Daily AI evolution cycle.

    Software Factory: The system continuously improves itself
    by discovering, benchmarking, and learning from better models.
    """
    import asyncio
    asyncio.run(_do_evolve())


async def _do_evolve():
    """Async implementation of the evolution cycle."""
    from app.services.intelligence.evolution_engine import get_evolution_engine

    engine = get_evolution_engine()
    result = await engine.evolve()

    if result["upgraded"]:
        logger.info(
            f"🚀 AI EVOLUTION: Upgrade recommended! "
            f"{result['current_model']} → {result['new_model']}"
        )
    else:
        logger.info(
            f"AI Evolution: No upgrade needed. Current model quality: "
            f"{result['current_benchmark']['quality']}"
        )

    logger.info(f"Self-teaching generated {result['self_teach_examples']} new training examples")
