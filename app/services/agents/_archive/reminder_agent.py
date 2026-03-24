"""Reminder Agent — Manages reminders and follow-ups."""
from app.services.agents.base import BaseAgent


class ReminderAgent(BaseAgent):
    identifier = "reminder_agent"
