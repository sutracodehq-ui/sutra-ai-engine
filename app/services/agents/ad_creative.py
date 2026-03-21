"""Ad Creative Agent — ad copy, hooks, CTAs for paid advertising."""
from app.services.agents.base import BaseAgent


class AdCreativeAgent(BaseAgent):
    @property
    def identifier(self) -> str:
        return "ad_creative"
