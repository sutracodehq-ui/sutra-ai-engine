import logging
from typing import Any, Optional

from app.services.intelligence.a2a_gateway import get_a2a_gateway

logger = logging.getLogger(__name__)

class AgentOrchestrator:
    """
    High-level orchestrator for complex multi-agent workflows.
    
    Provides a simplified interface to the A2A Gateway's pipeline patterns.
    """

    def __init__(self):
        self._gateway = get_a2a_gateway()

    async def execute_sequential(self, agents: list[str], prompt: str, db=None) -> dict:
        """Run a set of agents in sequence."""
        return await self._gateway.pipeline_sequential("orchestrator", agents, prompt, db=db)

    async def execute_brainstorm(self, analysts: list[str], strategist: str, prompt: str, db=None) -> dict:
        """Run a Map-Reduce pipeline for brainstorming."""
        return await self._gateway.pipeline_map_reduce("orchestrator", analysts, strategist, prompt, db=db)

    async def execute_refined_generation(self, producer: str, reviewer: str, prompt: str, max_iterations: int = 2, db=None) -> dict:
        """Run a Review-Chain pipeline for high-quality generation."""
        return await self._gateway.pipeline_review_chain("orchestrator", producer, reviewer, prompt, max_iterations, db=db)

# Singleton
_orchestrator = None

def get_agent_orchestrator() -> AgentOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = AgentOrchestrator()
    return _orchestrator
