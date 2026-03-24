"""Stock Predictor Agent — Probabilistic price forecasting."""
from app.services.agents.base import BaseAgent


class StockPredictorAgent(BaseAgent):
    identifier = "stock_predictor"
