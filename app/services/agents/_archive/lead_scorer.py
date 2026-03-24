"""LeadScorerAgent — scores leads and predicts conversion likelihood."""
from app.services.agents.base import BaseAgent

class LeadScorerAgent(BaseAgent):
    @property
    def identifier(self) -> str:
        return "lead_scorer"
