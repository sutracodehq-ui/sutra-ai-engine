"""Returns Predictor Agent — Predicts and reduces returns."""
from app.services.agents.base import BaseAgent


class ReturnsPredictorAgent(BaseAgent):
    identifier = "returns_predictor"
