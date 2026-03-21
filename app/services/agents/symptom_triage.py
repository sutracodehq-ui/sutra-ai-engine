"""Symptom Triage Agent — Assesses symptoms and suggests urgency level."""
from app.services.agents.base import BaseAgent


class SymptomTriageAgent(BaseAgent):
    identifier = "symptom_triage"
