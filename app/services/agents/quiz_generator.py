"""QuizGeneratorAgent — generates MCQs, fill-in-blanks, and assessments."""
from app.services.agents.base import BaseAgent

class QuizGeneratorAgent(BaseAgent):
    @property
    def identifier(self) -> str:
        return "quiz_generator"
