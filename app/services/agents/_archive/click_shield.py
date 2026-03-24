"""ClickShieldAgent — analyzes ad traffic and reports on fraud."""

from app.services.agents.base import BaseAgent

class ClickShieldAgent(BaseAgent):
    """
    Agent responsible for analyzing click logs and providing 
    fraud insights to the advertisers.
    """

    @property
    def identifier(self) -> str:
        return "click_shield"
