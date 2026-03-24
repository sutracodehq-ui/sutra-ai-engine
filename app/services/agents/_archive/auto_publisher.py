"""AutoPublisherAgent — creates content publishing schedules and distribution strategies."""
from app.services.agents.base import BaseAgent

class AutoPublisherAgent(BaseAgent):
    @property
    def identifier(self) -> str:
        return "auto_publisher"
