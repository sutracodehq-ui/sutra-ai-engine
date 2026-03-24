"""ML Pipeline Agent — Autonomous ML experiment runner."""
from app.services.agents.base import BaseAgent


class MlPipelineAgent(BaseAgent):
    identifier = "ml_pipeline"
