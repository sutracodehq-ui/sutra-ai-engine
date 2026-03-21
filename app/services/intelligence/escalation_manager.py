"""
Escalation Manager — Auto-escalate unknown queries to brand owner.

Software Factory Principle: Quality Control + Continuous Improvement.

When the chatbot AI isn't confident about an answer:
1. Detects low confidence (< threshold)
2. Sends WhatsApp notification to brand owner
3. Owner replies with the answer
4. AI learns the answer → responds to customer → knows it next time

Architecture:
    Low confidence response
        → EscalationManager.escalate()
            → Send WhatsApp to brand owner
            → Store pending escalation
        → Owner replies (via webhook)
            → ChatbotEngine.learn_from_owner()
            → BrandKnowledge.learn()
            → Reply to customer
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import httpx
import yaml

from app.config import get_settings

logger = logging.getLogger(__name__)


def _load_escalation_config() -> dict:
    """Load escalation config from YAML."""
    config_path = Path("intelligence_config.yaml")
    if not config_path.exists():
        return {}
    with open(config_path) as f:
        config = yaml.safe_load(f) or {}
    return config.get("chatbot", {}).get("escalation", {})


class PendingEscalation:
    """Tracks an escalated question waiting for owner response."""

    def __init__(
        self, escalation_id: str, brand_id: str, session_id: str,
        question: str, ai_response: str, confidence: float,
    ):
        self.escalation_id = escalation_id
        self.brand_id = brand_id
        self.session_id = session_id
        self.question = question
        self.ai_response = ai_response
        self.confidence = confidence
        self.created_at = datetime.now(timezone.utc)
        self.resolved = False
        self.owner_answer: str | None = None

    def to_dict(self) -> dict:
        return {
            "escalation_id": self.escalation_id,
            "brand_id": self.brand_id,
            "session_id": self.session_id,
            "question": self.question,
            "ai_response": self.ai_response[:200],
            "confidence": self.confidence,
            "resolved": self.resolved,
            "created_at": self.created_at.isoformat(),
        }


class EscalationManager:
    """
    Manages escalation of unknown queries to brand owners.

    Supports multiple notification channels:
    - WhatsApp (via Twilio/custom API)
    - Email (future)
    - SMS (future)
    - Push notification (future)
    """

    def __init__(self):
        self._pending: dict[str, PendingEscalation] = {}
        self._client = httpx.AsyncClient(timeout=30)
        self._log_dir = Path("training/escalations")
        self._log_dir.mkdir(parents=True, exist_ok=True)

    async def escalate(
        self,
        brand_id: str,
        session_id: str,
        customer_message: str,
        ai_response: str,
        confidence: float,
    ) -> bool:
        """
        Escalate an unknown query to the brand owner.
        Returns True if escalation was sent.
        """
        config = _load_escalation_config()
        threshold = config.get("confidence_threshold", 0.6)

        # Only escalate if confidence is below threshold
        if confidence >= threshold:
            return False

        # Generate escalation ID
        import hashlib
        esc_id = hashlib.md5(
            f"{brand_id}:{session_id}:{customer_message}:{datetime.now().isoformat()}".encode()
        ).hexdigest()[:12]

        # Store pending escalation
        escalation = PendingEscalation(
            escalation_id=esc_id,
            brand_id=brand_id,
            session_id=session_id,
            question=customer_message,
            ai_response=ai_response,
            confidence=confidence,
        )
        self._pending[esc_id] = escalation

        # Send WhatsApp notification
        sent = await self._send_whatsapp(escalation, config)

        # Log escalation
        self._log_escalation(escalation, sent)

        if sent:
            logger.info(
                f"Escalation: sent to owner for brand {brand_id} "
                f"(confidence={confidence:.2f}, esc_id={esc_id})"
            )

        return sent

    async def _send_whatsapp(self, escalation: PendingEscalation, config: dict) -> bool:
        """
        Send WhatsApp message to brand owner.
        Uses the configured WhatsApp API provider.
        """
        settings = get_settings()
        whatsapp_config = config.get("whatsapp", {})

        # Get brand owner's phone (from brand settings or config)
        # In production, this would come from the brand's database record
        owner_phone = whatsapp_config.get("default_owner_phone")
        api_url = whatsapp_config.get("api_url")
        api_key = whatsapp_config.get("api_key")

        if not all([owner_phone, api_url]):
            logger.warning("Escalation: WhatsApp not configured")
            return False

        # Format message
        message = (
            f"🤖 *Customer Query Alert*\n\n"
            f"A customer asked something I'm not sure about:\n\n"
            f"❓ *Question:*\n{escalation.question}\n\n"
            f"🤖 *My answer (confidence: {escalation.confidence:.0%}):*\n"
            f"{escalation.ai_response[:300]}\n\n"
            f"Please reply with the correct answer. "
            f"I'll learn it and respond to the customer.\n\n"
            f"_Reference: {escalation.escalation_id}_"
        )

        try:
            resp = await self._client.post(
                api_url,
                json={
                    "phone": owner_phone,
                    "message": message,
                    "escalation_id": escalation.escalation_id,
                },
                headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
            )
            return resp.status_code in (200, 201)
        except Exception as e:
            logger.error(f"Escalation: WhatsApp send failed: {e}")
            return False

    async def resolve(self, escalation_id: str, owner_answer: str) -> dict:
        """
        Resolve an escalation with the brand owner's answer.
        Called when owner replies via WhatsApp webhook.
        """
        escalation = self._pending.get(escalation_id)
        if not escalation:
            return {"status": "not_found", "escalation_id": escalation_id}

        escalation.resolved = True
        escalation.owner_answer = owner_answer

        # Learn the answer via chatbot engine
        from app.services.intelligence.chatbot_engine import get_chatbot_engine
        engine = get_chatbot_engine()

        result = await engine.learn_from_owner(
            brand_id=escalation.brand_id,
            question=escalation.question,
            answer=owner_answer,
            session_id=escalation.session_id,
        )

        # Remove from pending
        del self._pending[escalation_id]

        logger.info(f"Escalation: resolved {escalation_id} for brand {escalation.brand_id}")

        return {
            "status": "resolved",
            "escalation_id": escalation_id,
            "brand_id": escalation.brand_id,
            "question": escalation.question[:100],
            "learned": True,
        }

    def get_pending(self, brand_id: str | None = None) -> list[dict]:
        """Get all pending escalations, optionally filtered by brand."""
        pending = []
        for esc in self._pending.values():
            if brand_id and esc.brand_id != brand_id:
                continue
            if not esc.resolved:
                pending.append(esc.to_dict())
        return pending

    def _log_escalation(self, escalation: PendingEscalation, sent: bool):
        """Log escalation for analytics."""
        log_path = self._log_dir / f"escalations_{escalation.brand_id}.jsonl"
        entry = {**escalation.to_dict(), "notification_sent": sent}
        try:
            with open(log_path, "a") as f:
                f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
        except Exception:
            pass

    async def close(self):
        await self._client.aclose()


# ─── Singleton ──────────────────────────────────────────────
_manager: EscalationManager | None = None


def get_escalation_manager() -> EscalationManager:
    global _manager
    if _manager is None:
        _manager = EscalationManager()
    return _manager
