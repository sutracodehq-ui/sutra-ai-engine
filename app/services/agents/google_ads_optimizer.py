"""GoogleAdsOptimizerAgent — optimizes Google Ads campaigns."""
from app.services.agents.base import BaseAgent

class GoogleAdsOptimizerAgent(BaseAgent):
    @property
    def identifier(self) -> str:
        return "google_ads_optimizer"
