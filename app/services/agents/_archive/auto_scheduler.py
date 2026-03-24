"""AutoSchedulerAgent — finds optimal posting times and generates schedules."""
from app.services.agents.base import BaseAgent

class AutoSchedulerAgent(BaseAgent):
    @property
    def identifier(self) -> str:
        return "auto_scheduler"
