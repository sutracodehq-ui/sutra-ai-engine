"""ReviewReputationAgent — monitors and manages online reviews."""
from app.services.agents.base import BaseAgent

class ReviewReputationAgent(BaseAgent):
    @property
    def identifier(self) -> str:
        return "review_reputation"
