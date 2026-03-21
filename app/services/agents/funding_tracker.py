"""FundingTrackerAgent — startup domain agent."""
from app.services.agents.base import BaseAgent


class FundingTrackerAgent(BaseAgent):
    identifier = "funding_tracker"
