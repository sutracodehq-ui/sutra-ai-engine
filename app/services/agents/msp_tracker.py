"""MSP Tracker Agent — Crop price and mandi tracking."""
from app.services.agents.base import BaseAgent


class MspTrackerAgent(BaseAgent):
    identifier = "msp_tracker"
