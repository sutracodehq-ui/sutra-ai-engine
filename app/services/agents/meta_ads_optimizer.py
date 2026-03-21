"""MetaAdsOptimizerAgent — optimizes Facebook and Instagram ad campaigns."""
from app.services.agents.base import BaseAgent

class MetaAdsOptimizerAgent(BaseAgent):
    @property
    def identifier(self) -> str:
        return "meta_ads_optimizer"
