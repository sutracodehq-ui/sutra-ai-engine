"""AudioDubberAgent — translates transcripts and prepares for TTS dubbing."""
from app.services.agents.base import BaseAgent

class AudioDubberAgent(BaseAgent):
    @property
    def identifier(self) -> str:
        return "audio_dubber"
