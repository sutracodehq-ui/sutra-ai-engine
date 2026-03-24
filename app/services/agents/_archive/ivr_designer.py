"""IvrDesignerAgent — designs IVR menu flows with multilingual support."""
from app.services.agents.base import BaseAgent

class IvrDesignerAgent(BaseAgent):
    @property
    def identifier(self) -> str:
        return "ivr_designer"
