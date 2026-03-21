"""Meeting Notes Agent — Extracts decisions and action items."""
from app.services.agents.base import BaseAgent


class MeetingNotesAgent(BaseAgent):
    identifier = "meeting_notes"
