"""Review Analyzer Agent — Product review sentiment analysis."""
from app.services.agents.base import BaseAgent


class ReviewAnalyzerAgent(BaseAgent):
    identifier = "review_analyzer"
