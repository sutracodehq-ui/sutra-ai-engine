"""SocialClipMakerAgent — identifies viral moments for Reels/Shorts."""
from app.services.agents.base import BaseAgent

class SocialClipMakerAgent(BaseAgent):
    @property
    def identifier(self) -> str:
        return "social_clip_maker"
