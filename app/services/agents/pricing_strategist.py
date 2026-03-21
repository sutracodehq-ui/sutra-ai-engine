"""PricingStrategistAgent — analyses competitive pricing and recommends strategies."""
from app.services.agents.base import BaseAgent

class PricingStrategistAgent(BaseAgent):
    @property
    def identifier(self) -> str:
        return "pricing_strategist"
