"""PasswordAuditorAgent — cybersecurity domain agent."""
from app.services.agents.base import BaseAgent


class PasswordAuditorAgent(BaseAgent):
    identifier = "password_auditor"
