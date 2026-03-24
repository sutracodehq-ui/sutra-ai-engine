"""NoteGeneratorAgent — generates structured study notes from lectures."""
from app.services.agents.base import BaseAgent

class NoteGeneratorAgent(BaseAgent):
    @property
    def identifier(self) -> str:
        return "note_generator"
