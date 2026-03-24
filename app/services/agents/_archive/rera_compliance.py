"""RERA Compliance Agent — Real estate regulation checks."""
from app.services.agents.base import BaseAgent


class ReraComplianceAgent(BaseAgent):
    identifier = "rera_compliance"
