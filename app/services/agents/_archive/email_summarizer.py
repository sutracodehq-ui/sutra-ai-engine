"""Email Summarizer Agent — Summarizes threads and extracts actions."""
from app.services.agents.base import BaseAgent


class EmailSummarizerAgent(BaseAgent):
    identifier = "email_summarizer"
