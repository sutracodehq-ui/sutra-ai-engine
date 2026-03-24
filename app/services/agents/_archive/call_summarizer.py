"""CallSummarizerAgent — summarizes call transcriptions into actionable notes."""
from app.services.agents.base import BaseAgent

class CallSummarizerAgent(BaseAgent):
    @property
    def identifier(self) -> str:
        return "call_summarizer"
