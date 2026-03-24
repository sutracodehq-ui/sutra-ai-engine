"""Contract Analyzer Agent — Analyzes contracts and identifies risks."""
from app.services.agents.base import BaseAgent


class ContractAnalyzerAgent(BaseAgent):
    identifier = "contract_analyzer"
