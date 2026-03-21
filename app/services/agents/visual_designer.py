"""VisualDesignerAgent — generates design briefs and image prompts."""
from app.services.agents.base import BaseAgent

class VisualDesignerAgent(BaseAgent):
    @property
    def identifier(self) -> str:
        return "visual_designer"
