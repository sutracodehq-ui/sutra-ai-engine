"""CompetitorAnalystAgent — competitive intelligence and SWOT analysis."""
from app.services.agents.base import BaseAgent

class CompetitorAnalystAgent(BaseAgent):
    @property
    def identifier(self) -> str:
        return "competitor_analyst"
