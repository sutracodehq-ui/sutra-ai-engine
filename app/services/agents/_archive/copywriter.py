"""Copywriter Agent — headlines, body copy, CTAs."""
from app.services.agents.base import BaseAgent


class CopywriterAgent(BaseAgent):
    @property
    def identifier(self) -> str:
        return "copywriter"
