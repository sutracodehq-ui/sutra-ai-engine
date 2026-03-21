"""FunnelAnalyzerAgent — analyzes marketing funnel and identifies bottlenecks."""
from app.services.agents.base import BaseAgent

class FunnelAnalyzerAgent(BaseAgent):
    @property
    def identifier(self) -> str:
        return "funnel_analyzer"
