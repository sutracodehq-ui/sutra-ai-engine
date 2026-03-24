"""Resume Screener Agent — Screens and scores resumes."""
from app.services.agents.base import BaseAgent


class ResumeScreenerAgent(BaseAgent):
    identifier = "resume_screener"
