"""LecturePlannerAgent — plans full lecture series from topics or syllabi."""
from app.services.agents.base import BaseAgent

class LecturePlannerAgent(BaseAgent):
    @property
    def identifier(self) -> str:
        return "lecture_planner"
