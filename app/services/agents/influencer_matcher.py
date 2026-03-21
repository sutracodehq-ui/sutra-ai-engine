"""InfluencerMatcherAgent — finds ideal influencers for brand partnerships."""
from app.services.agents.base import BaseAgent

class InfluencerMatcherAgent(BaseAgent):
    @property
    def identifier(self) -> str:
        return "influencer_matcher"
