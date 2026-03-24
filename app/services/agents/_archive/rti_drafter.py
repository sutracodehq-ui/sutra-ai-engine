"""RTI Drafter Agent — Drafts Right to Information applications."""
from app.services.agents.base import BaseAgent


class RtiDrafterAgent(BaseAgent):
    identifier = "rti_drafter"
