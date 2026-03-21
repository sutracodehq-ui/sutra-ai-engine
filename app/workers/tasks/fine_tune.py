"""
LoRA Fine-Tune + Feedback Export — Weekly Celery task.

Runs every Sunday at 5 AM:
1. Export positive user feedback as JSONL training data
2. Run the LoRA fine-tuning pipeline with all accumulated data
"""

import logging

from app.workers.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, name="fine_tune_model", max_retries=1)
def fine_tune_model(self):
    """
    Weekly LoRA fine-tuning cycle.

    Software Factory: The system continuously improves via user feedback
    and self-generated training data.
    """
    import asyncio
    asyncio.run(_do_fine_tune())


async def _do_fine_tune():
    """Async implementation of fine-tuning pipeline."""
    # 1. Export positive user feedback
    try:
        from app.db.session import async_session_factory
        from app.services.intelligence.feedback_collector import FeedbackCollector

        async with async_session_factory() as db:
            collector = FeedbackCollector(db)
            export_result = await collector.export_positive_examples()
            logger.info(f"Feedback export: {export_result}")
    except Exception as e:
        logger.warning(f"Feedback export skipped: {e}")

    # 2. Run LoRA fine-tuning
    from app.services.intelligence.lora_trainer import get_lora_trainer

    trainer = get_lora_trainer()
    result = await trainer.run_pipeline()

    if result["status"] == "success":
        logger.info(
            f"🎓 Fine-tuning complete: {result['base_model']} → {result['new_model']} "
            f"({result['examples_used']} examples)"
        )
    elif result["status"] == "skipped":
        logger.info(f"Fine-tuning skipped: {result['reason']}")
    else:
        logger.warning(f"Fine-tuning failed: {result.get('error', 'unknown')}")
