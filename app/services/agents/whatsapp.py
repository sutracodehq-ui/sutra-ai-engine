"""WhatsApp Agent — message templates, conversational flows."""
from app.services.agents.base import BaseAgent


class WhatsappAgent(BaseAgent):
    @property
    def identifier(self) -> str:
        return "whatsapp"
