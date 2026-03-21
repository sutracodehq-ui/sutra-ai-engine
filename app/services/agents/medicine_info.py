"""Medicine Info Agent — Drug information, interactions, and Indian brand mapping."""
from app.services.agents.base import BaseAgent


class MedicineInfoAgent(BaseAgent):
    identifier = "medicine_info"
