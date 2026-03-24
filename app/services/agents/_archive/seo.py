"""SEO Agent — meta tags, keywords, content optimization."""
from app.services.agents.base import BaseAgent


class SeoAgent(BaseAgent):
    @property
    def identifier(self) -> str:
        return "seo"
