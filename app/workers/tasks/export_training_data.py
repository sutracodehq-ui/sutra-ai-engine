"""
Export Training Data — Celery periodic task.

Self-Learning Tier 2: Runs weekly to collect feedback-rated agent
interactions and export them as JSONL for fine-tuning.

Schedule: Every Sunday at 2:00 AM (configurable via Celery Beat).
"""

import logging
from datetime import datetime, timedelta

from app.workers.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, name="export_training_data", max_retries=1)
def export_training_data(self, days_back: int = 7):
    """
    Export the last N days of feedback-rated responses as training data.

    Args:
        days_back: Number of days to look back for new feedback (default: 7).
    """
    import asyncio
    asyncio.run(_do_export(days_back))


async def _do_export(days_back: int):
    """Async implementation of the training data export."""
    from app.db.session import async_session_factory
    from app.services.intelligence.training_collector import TrainingDataCollector

    since = datetime.utcnow() - timedelta(days=days_back)

    async with async_session_factory() as db:
        collector = TrainingDataCollector(db)
        result = await collector.export_jsonl(since=since)

    if result["total_examples"] > 0:
        logger.info(
            f"Training data exported: {result['total_examples']} examples "
            f"to {result['path']}, breakdown: {result['by_agent']}"
        )
    else:
        logger.info("No new training data to export this cycle.")
