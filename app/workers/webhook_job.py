"""
Webhook Job — async worker for connectivity tasks.
"""

import logging
from app.workers.celery_app import celery_app
from app.services.connectivity.webhooks import WebhookService

logger = logging.getLogger(__name__)


@celery_app.task(name="app.workers.webhook_job.dispatch_webhook")
async def dispatch_webhook(url: str, payload: dict, event_type: str):
    """Celery task to dispatch a webhook without blocking the main flow."""
    await WebhookService.dispatch(url, payload, event_type)


@celery_app.task(name="app.workers.webhook_job.trigger_frustration_alert")
async def trigger_frustration_alert(tenant_id: int, sentiment: dict, webhook_url: str):
    """Celery task to trigger a frustration alert."""
    await WebhookService.alert_frustration(tenant_id, sentiment, webhook_url)
