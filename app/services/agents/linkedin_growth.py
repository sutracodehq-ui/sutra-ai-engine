"""LinkedinGrowthAgent — generates authority-building LinkedIn content and B2B outreach."""
from app.services.agents.base import BaseAgent

class LinkedinGrowthAgent(BaseAgent):
    @property
    def identifier(self) -> str:
        return "linkedin_growth"
