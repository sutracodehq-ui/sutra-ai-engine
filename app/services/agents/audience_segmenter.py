"""AudienceSegmenterAgent — creates micro-segments with personalized messaging."""
from app.services.agents.base import BaseAgent

class AudienceSegmenterAgent(BaseAgent):
    @property
    def identifier(self) -> str:
        return "audience_segmenter"
