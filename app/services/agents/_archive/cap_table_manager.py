"""CapTableManagerAgent — startup domain agent."""
from app.services.agents.base import BaseAgent


class CapTableManagerAgent(BaseAgent):
    identifier = "cap_table_manager"
