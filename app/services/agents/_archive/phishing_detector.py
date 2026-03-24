"""PhishingDetectorAgent — cybersecurity domain agent."""
from app.services.agents.base import BaseAgent


class PhishingDetectorAgent(BaseAgent):
    identifier = "phishing_detector"
