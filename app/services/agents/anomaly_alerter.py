"""AnomalyAlerterAgent — detects anomalies in marketing metrics."""
from app.services.agents.base import BaseAgent

class AnomalyAlerterAgent(BaseAgent):
    @property
    def identifier(self) -> str:
        return "anomaly_alerter"
