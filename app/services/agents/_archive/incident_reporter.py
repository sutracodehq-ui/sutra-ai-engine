"""IncidentReporterAgent — cybersecurity domain agent."""
from app.services.agents.base import BaseAgent


class IncidentReporterAgent(BaseAgent):
    identifier = "incident_reporter"
