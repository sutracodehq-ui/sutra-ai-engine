"""Interview Question Generator Agent — Creates role-specific questions."""
from app.services.agents.base import BaseAgent


class InterviewQGeneratorAgent(BaseAgent):
    identifier = "interview_q_generator"
