"""JourneyMapperAgent — maps customer touchpoints across channels."""
from app.services.agents.base import BaseAgent

class JourneyMapperAgent(BaseAgent):
    @property
    def identifier(self) -> str:
        return "journey_mapper"
