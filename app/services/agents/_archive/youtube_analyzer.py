"""YoutubeAnalyzerAgent — analyzes YouTube videos and R2-hosted media."""
from app.services.agents.base import BaseAgent

class YoutubeAnalyzerAgent(BaseAgent):
    @property
    def identifier(self) -> str:
        return "youtube_analyzer"
