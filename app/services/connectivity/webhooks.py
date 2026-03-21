"""
Webhook Service — handles external connectivity and alerts.

Connectivity-AI: Dispatches async notifications to third-party systems 
(Slack, Discord, Custom APIs).
"""

import logging
import httpx
from typing import Any, Dict

from app.config import get_settings

logger = logging.getLogger(__name__)


class WebhookService:
    """Service to dispatch asynchronous webhooks."""

    @classmethod
    async def dispatch(cls, url: str, payload: Dict[str, Any], event_type: str):
        """Send a POST request to the specified webhook URL."""
        if not url:
            return

        logger.info(f"Dispatching '{event_type}' webhook to {url}")
        
        # Add metadata
        payload["event"] = event_type
        
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                logger.info(f"Webhook '{event_type}' delivered successfully.")
        except Exception as e:
            logger.error(f"Webhook dispatch failed: {e}")

    @classmethod
    async def alert_frustration(cls, tenant_id: int, sentiment: dict, webhook_url: str | None):
        """Specialized alert for high user frustration."""
        if not webhook_url:
            return

        payload = {
            "tenant_id": tenant_id,
            "sentiment_label": sentiment.get("label"),
            "sentiment_score": sentiment.get("score"),
            "vibe": sentiment.get("vibe"),
            "alert_level": "critical" if sentiment.get("label") == "angry" else "high",
            "message": "User frustration detected. Intervention suggested."
        }
        
        await cls.dispatch(webhook_url, payload, "customer_frustration_alert")
