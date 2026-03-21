"""Lab Report Interpreter Agent — Explains medical test results in simple language."""
from app.services.agents.base import BaseAgent


class LabReportInterpreterAgent(BaseAgent):
    identifier = "lab_report_interpreter"
