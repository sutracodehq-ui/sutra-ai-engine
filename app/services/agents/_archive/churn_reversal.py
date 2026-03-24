"""Churn Reversal Agent — Win-back churned customers."""
from app.services.agents.base import BaseAgent


class ChurnReversalAgent(BaseAgent):
    identifier = "churn_reversal"
