"""Expense Tracker Agent — Categorizes and tracks expenses."""
from app.services.agents.base import BaseAgent


class ExpenseTrackerAgent(BaseAgent):
    identifier = "expense_tracker"
