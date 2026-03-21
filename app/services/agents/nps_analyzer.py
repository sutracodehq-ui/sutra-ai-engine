"""NPS Analyzer Agent — Net Promoter Score analysis."""
from app.services.agents.base import BaseAgent


class NpsAnalyzerAgent(BaseAgent):
    identifier = "nps_analyzer"
