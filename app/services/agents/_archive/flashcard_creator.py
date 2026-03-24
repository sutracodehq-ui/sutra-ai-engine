"""FlashcardCreatorAgent — creates spaced-repetition flashcards."""
from app.services.agents.base import BaseAgent

class FlashcardCreatorAgent(BaseAgent):
    @property
    def identifier(self) -> str:
        return "flashcard_creator"
