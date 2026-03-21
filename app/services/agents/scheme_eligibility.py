"""Scheme Eligibility Agent — Government scheme eligibility check."""
from app.services.agents.base import BaseAgent


class SchemeEligibilityAgent(BaseAgent):
    identifier = "scheme_eligibility"
