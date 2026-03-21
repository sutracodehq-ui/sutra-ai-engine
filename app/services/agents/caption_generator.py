"""CaptionGeneratorAgent — generates SRT/VTT subtitles in any Indian language."""
from app.services.agents.base import BaseAgent

class CaptionGeneratorAgent(BaseAgent):
    @property
    def identifier(self) -> str:
        return "caption_generator"
