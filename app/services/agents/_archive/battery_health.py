"""BatteryHealthAgent — ev_green_energy domain agent."""
from app.services.agents.base import BaseAgent


class BatteryHealthAgent(BaseAgent):
    identifier = "battery_health"
