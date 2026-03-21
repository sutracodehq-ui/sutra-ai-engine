"""CallSentimentAnalyzerAgent — analyzes call recordings for sentiment and insights."""
from app.services.agents.base import BaseAgent

class CallSentimentAnalyzerAgent(BaseAgent):
    @property
    def identifier(self) -> str:
        return "call_sentiment_analyzer"
