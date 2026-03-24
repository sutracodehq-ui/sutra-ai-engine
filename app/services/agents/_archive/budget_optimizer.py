"""BudgetOptimizerAgent — recommends optimal marketing budget allocation."""
from app.services.agents.base import BaseAgent

class BudgetOptimizerAgent(BaseAgent):
    @property
    def identifier(self) -> str:
        return "budget_optimizer"
