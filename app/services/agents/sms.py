"""SMS Agent — short message copy, variants, compliance."""
from app.services.agents.base import BaseAgent


class SmsAgent(BaseAgent):
    @property
    def identifier(self) -> str:
        return "sms"
