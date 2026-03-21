"""ChatbotTrainerAgent — generates FAQ responses and conversational AI training data."""
from app.services.agents.base import BaseAgent

class ChatbotTrainerAgent(BaseAgent):
    @property
    def identifier(self) -> str:
        return "chatbot_trainer"
