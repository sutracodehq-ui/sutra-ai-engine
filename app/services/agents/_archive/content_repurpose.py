"""Content Repurposer Agent — multi-channel adaptation."""
from app.services.agents.base import BaseAgent


class ContentRepurposerAgent(BaseAgent):
    @property
    def identifier(self) -> str:
        return "content_repurpose"
