"""PetHealthAgent — pet_care domain agent."""
from app.services.agents.base import BaseAgent


class PetHealthAgent(BaseAgent):
    identifier = "pet_health"
