"""FakeNewsDetectorAgent — media domain agent."""
from app.services.agents.base import BaseAgent


class FakeNewsDetectorAgent(BaseAgent):
    identifier = "fake_news_detector"
