"""PersonaBuilderAgent — generates buyer personas and audience segments."""
from app.services.agents.base import BaseAgent

class PersonaBuilderAgent(BaseAgent):
    @property
    def identifier(self) -> str:
        return "persona_builder"
