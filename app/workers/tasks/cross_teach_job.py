"""
Cross-Teach Job — Legacy Stub.

Formerly transferred skills between agents.
Features now managed by unified Knowledge Graph and RAG in Memory.
"""

import logging
from app.workers.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, name="cross_teach_job", max_retries=1)
def cross_teach_job(self):
    """Legacy cross-teach worker redirect."""
    logger.info("Cross-teach task completed: logic absorbed into unified Memory engine.")
