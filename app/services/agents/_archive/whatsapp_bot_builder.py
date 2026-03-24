"""WhatsappBotBuilderAgent — designs complete WhatsApp Business bot flows."""
from app.services.agents.base import BaseAgent

class WhatsappBotBuilderAgent(BaseAgent):
    @property
    def identifier(self) -> str:
        return "whatsapp_bot_builder"
