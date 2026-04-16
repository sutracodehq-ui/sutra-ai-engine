"""
Web Intelligence Scan — Celery periodic task.

Runs every hour to scan RSS feeds, stock prices, and crypto markets.
Stores everything in Qdrant for agent context injection.
"""

import logging

from app.workers.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, name="scan_web_intelligence", max_retries=2)
def scan_web_intelligence(self):
    """
    Hourly web intelligence scan.

    Fetches latest AI trends, marketing news, stock prices, and crypto data.
    All data stored in Qdrant for instant retrieval by agents.
    """
    import asyncio
    asyncio.run(_do_scan())


async def _do_scan():
    """Async implementation of the web scan via Memory."""
    from app.services.intelligence.memory import get_memory

    mem = get_memory()
    result = await mem.full_scan()

    logger.info(
        f"Web scan complete: articles={result['articles']}, "
        f"stocks={result['stocks']}, crypto={result['crypto']}"
    )

    if result["errors"]:
        logger.warning(f"Web scan errors: {result['errors']}")
