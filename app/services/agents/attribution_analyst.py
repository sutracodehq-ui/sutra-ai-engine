"""AttributionAnalystAgent — multi-channel attribution modeling."""
from app.services.agents.base import BaseAgent

class AttributionAnalystAgent(BaseAgent):
    @property
    def identifier(self) -> str:
        return "attribution_analyst"
