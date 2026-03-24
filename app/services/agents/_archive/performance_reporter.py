"""PerformanceReporterAgent — summarizes marketing KPIs and trends."""
from app.services.agents.base import BaseAgent

class PerformanceReporterAgent(BaseAgent):
    @property
    def identifier(self) -> str:
        return "performance_reporter"
