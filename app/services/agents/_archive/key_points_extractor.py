"""KeyPointsExtractorAgent — extracts formulas, definitions, theorems from content."""
from app.services.agents.base import BaseAgent

class KeyPointsExtractorAgent(BaseAgent):
    @property
    def identifier(self) -> str:
        return "key_points_extractor"
