"""LandingPageBuilderAgent — generates high-converting landing page copy."""
from app.services.agents.base import BaseAgent

class LandingPageBuilderAgent(BaseAgent):
    @property
    def identifier(self) -> str:
        return "landing_page_builder"
