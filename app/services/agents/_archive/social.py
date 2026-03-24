"""Social Media Agent — platform-optimized posts, hashtags, image prompts."""
from app.services.agents.base import BaseAgent


class SocialAgent(BaseAgent):
    @property
    def identifier(self) -> str:
        return "social_media"
