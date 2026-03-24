"""ColdCallScripterAgent — generates persuasive cold call scripts with objection handling."""
from app.services.agents.base import BaseAgent

class ColdCallScripterAgent(BaseAgent):
    @property
    def identifier(self) -> str:
        return "cold_call_scripter"
