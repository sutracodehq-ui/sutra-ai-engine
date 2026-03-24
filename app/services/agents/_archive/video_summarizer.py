"""VideoSummarizerAgent — summarizes long videos into chapters and key moments."""
from app.services.agents.base import BaseAgent

class VideoSummarizerAgent(BaseAgent):
    @property
    def identifier(self) -> str:
        return "video_summarizer"
