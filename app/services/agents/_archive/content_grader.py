"""ContentGraderAgent — scores content quality across multiple dimensions."""
from app.services.agents.base import BaseAgent

class ContentGraderAgent(BaseAgent):
    @property
    def identifier(self) -> str:
        return "content_grader"
