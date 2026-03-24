"""CampaignStrategistAgent — designs multi-channel campaign blueprints."""
from app.services.agents.base import BaseAgent

class CampaignStrategistAgent(BaseAgent):
    @property
    def identifier(self) -> str:
        return "campaign_strategist"
