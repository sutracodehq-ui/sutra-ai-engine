"""Brand Auditor Agent — consistency and style guide auditing."""
from app.services.agents.base import BaseAgent


class BrandAuditorAgent(BaseAgent):
    @property
    def identifier(self) -> str:
        return "brand_auditor"
