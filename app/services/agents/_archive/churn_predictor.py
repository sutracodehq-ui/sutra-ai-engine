"""ChurnPredictorAgent — predicts customer churn and suggests retention strategies."""
from app.services.agents.base import BaseAgent

class ChurnPredictorAgent(BaseAgent):
    @property
    def identifier(self) -> str:
        return "churn_predictor"
