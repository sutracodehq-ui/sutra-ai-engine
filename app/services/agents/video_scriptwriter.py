"""VideoScriptwriterAgent — writes scripts for reels, ads, and explainers."""
from app.services.agents.base import BaseAgent

class VideoScriptwriterAgent(BaseAgent):
    @property
    def identifier(self) -> str:
        return "video_scriptwriter"
