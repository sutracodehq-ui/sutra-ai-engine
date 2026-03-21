"""Email Campaign Agent — subject lines, email copy, A/B variants."""
from app.services.agents.base import BaseAgent


class EmailCampaignAgent(BaseAgent):
    @property
    def identifier(self) -> str:
        return "email_campaign"
