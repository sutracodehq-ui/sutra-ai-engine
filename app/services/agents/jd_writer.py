"""JD Writer Agent — Creates job descriptions."""
from app.services.agents.base import BaseAgent


class JdWriterAgent(BaseAgent):
    identifier = "jd_writer"
