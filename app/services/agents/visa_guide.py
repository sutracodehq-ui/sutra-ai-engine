"""Visa Guide Agent — Visa requirements for Indian passport."""
from app.services.agents.base import BaseAgent


class VisaGuideAgent(BaseAgent):
    identifier = "visa_guide"
