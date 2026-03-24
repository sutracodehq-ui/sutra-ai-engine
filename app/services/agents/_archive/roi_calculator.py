"""RoiCalculatorAgent — calculates marketing ROI, ROAS, CAC, LTV."""
from app.services.agents.base import BaseAgent

class RoiCalculatorAgent(BaseAgent):
    @property
    def identifier(self) -> str:
        return "roi_calculator"
