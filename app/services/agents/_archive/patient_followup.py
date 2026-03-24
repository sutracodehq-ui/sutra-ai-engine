"""Patient Follow-up Agent — Post-visit care and medication reminders."""
from app.services.agents.base import BaseAgent


class PatientFollowupAgent(BaseAgent):
    identifier = "patient_followup"
