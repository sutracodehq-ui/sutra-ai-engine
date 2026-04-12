"""
Export Training Data / Fine-tune — Legacy Stub.

Self-Learning Tier 2: Formerly ran weekly to collect feedback-rated agent
interactions and export them as JSONL for fine-tuning.

Now handled by Memory.export_jsonl().
"""

import logging
from app.workers.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, name="fine_tune", max_retries=1)
def fine_tune(self):
    """Legacy fine-tune worker redirect."""
    import asyncio
    asyncio.run(_do_fine_tune())


async def _do_fine_tune():
    """Legacy fine-tune worker. Training is now handled by Memory.export_jsonl."""
    logger.info("Fine-tune task skipped: use Memory.export_jsonl for manual tuning.")
