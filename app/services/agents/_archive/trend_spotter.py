"""TrendSpotterAgent — identifies emerging trends and viral opportunities."""
from app.services.agents.base import BaseAgent

class TrendSpotterAgent(BaseAgent):
    @property
    def identifier(self) -> str:
        return "trend_spotter"
