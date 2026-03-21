"""AbTestAdvisorAgent — creates A/B test variations and analyzes results."""
from app.services.agents.base import BaseAgent

class AbTestAdvisorAgent(BaseAgent):
    @property
    def identifier(self) -> str:
        return "ab_test_advisor"
