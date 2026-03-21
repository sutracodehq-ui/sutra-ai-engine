"""
A2A Gateway — Agent-to-Agent Communication Protocol.

Software Factory Principle: Modular Architecture + Collaboration.

Enables agents to discover, consult, and delegate tasks to each other.
Inspired by Google's A2A Protocol (Linux Foundation).

Architecture:
    Agent A → A2A Gateway → Route lookup → Agent B.execute() → Result back to A

Key Concepts:
- Agent Card: Name, capabilities, allowed consultants (from YAML)
- Consult: Ask another agent for input (synchronous)
- Delegate: Hand off a sub-task to another agent (async-capable)
- Broadcast: Ask multiple agents and merge results
"""

import logging
from typing import Any, Optional

import yaml
from pathlib import Path

logger = logging.getLogger(__name__)

CONFIG_PATH = Path("intelligence_config.yaml")


def _load_a2a_routes() -> dict:
    """Load A2A routing config from YAML."""
    if not CONFIG_PATH.exists():
        return {}
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f) or {}
    return config.get("a2a_routes", {})


class AgentCard:
    """Describes an agent's identity and capabilities for discovery."""

    def __init__(self, identifier: str, name: str, domain: str, capabilities: list[str]):
        self.identifier = identifier
        self.name = name
        self.domain = domain
        self.capabilities = capabilities

    def to_dict(self) -> dict:
        return {
            "identifier": self.identifier,
            "name": self.name,
            "domain": self.domain,
            "capabilities": self.capabilities,
        }


class A2AGateway:
    """
    Agent-to-Agent communication gateway.

    Allows agents to:
    1. Discover other agents via Agent Cards
    2. Consult another agent (ask for input, get response)
    3. Delegate a sub-task (hand off work entirely)
    4. Broadcast to multiple agents and merge results
    """

    def __init__(self):
        self._hub = None  # Lazy-loaded

    def _get_hub(self):
        """Lazy-load the agent hub to avoid circular imports."""
        if self._hub is None:
            from app.services.agents.hub import AiAgentHub
            self._hub = AiAgentHub()
        return self._hub

    # ─── Discovery ──────────────────────────────────────────

    def discover(self, domain: str | None = None) -> list[AgentCard]:
        """
        Discover available agents, optionally filtered by domain.
        Returns Agent Cards describing each agent's capabilities.
        """
        hub = self._get_hub()
        cards = []
        for identifier, agent_class in hub._registry.items():
            agent = agent_class()
            info = agent.info()
            if domain and info.get("domain") != domain:
                continue
            cards.append(AgentCard(
                identifier=info["identifier"],
                name=info["name"],
                domain=info.get("domain", "general"),
                capabilities=info.get("capabilities", []),
            ))
        return cards

    def can_consult(self, source_agent: str, target_agent: str) -> bool:
        """Check if source agent is allowed to consult target agent."""
        routes = _load_a2a_routes()
        agent_routes = routes.get(source_agent, {})
        allowed = agent_routes.get("can_consult", []) + agent_routes.get("can_delegate", [])
        return target_agent in allowed or "*" in allowed

    # ─── Consult (synchronous ask) ──────────────────────────

    async def consult(
        self,
        source_agent: str,
        target_agent: str,
        prompt: str,
        context: dict | None = None,
        db=None,
    ) -> dict:
        """
        One agent consults another for input.

        Example: stock_predictor consults ai_trend_tracker for
        "Any recent AI developments affecting NVDA stock?"
        """
        if not self.can_consult(source_agent, target_agent):
            logger.warning(
                f"A2A: {source_agent} not allowed to consult {target_agent}"
            )
            return {"status": "denied", "reason": "Route not configured"}

        hub = self._get_hub()
        agent = hub.get(target_agent)
        if not agent:
            return {"status": "error", "reason": f"Agent {target_agent} not found"}

        try:
            # Inject A2A context
            a2a_context = {
                **(context or {}),
                "a2a_source": source_agent,
                "a2a_type": "consult",
            }

            response = await agent.execute(prompt, db=db, context=a2a_context)

            logger.info(f"A2A: {source_agent} → consult → {target_agent} ✓")
            return {
                "status": "success",
                "source": source_agent,
                "target": target_agent,
                "response": response.content,
                "metadata": response.metadata,
            }
        except Exception as e:
            logger.error(f"A2A: consult {target_agent} failed: {e}")
            return {"status": "error", "reason": str(e)}

    # ─── Delegate (hand off task) ───────────────────────────

    async def delegate(
        self,
        source_agent: str,
        target_agent: str,
        task: str,
        context: dict | None = None,
        db=None,
    ) -> dict:
        """
        One agent delegates a sub-task to another.

        Unlike consult, delegate implies the target agent
        has full ownership of producing the result.
        """
        if not self.can_consult(source_agent, target_agent):
            return {"status": "denied", "reason": "Route not configured"}

        hub = self._get_hub()
        agent = hub.get(target_agent)
        if not agent:
            return {"status": "error", "reason": f"Agent {target_agent} not found"}

        try:
            a2a_context = {
                **(context or {}),
                "a2a_source": source_agent,
                "a2a_type": "delegate",
            }

            response = await agent.execute(task, db=db, context=a2a_context)

            logger.info(f"A2A: {source_agent} → delegate → {target_agent} ✓")
            return {
                "status": "success",
                "source": source_agent,
                "target": target_agent,
                "response": response.content,
            }
        except Exception as e:
            logger.error(f"A2A: delegate to {target_agent} failed: {e}")
            return {"status": "error", "reason": str(e)}

    # ─── Broadcast (ask multiple agents) ────────────────────

    async def broadcast(
        self,
        source_agent: str,
        target_agents: list[str],
        prompt: str,
        context: dict | None = None,
        db=None,
    ) -> list[dict]:
        """
        Broadcast a query to multiple agents and collect all responses.

        Example: orchestrator broadcasts "Analyze NVDA" to
        [stock_analyzer, stock_predictor, ai_trend_tracker]
        """
        import asyncio

        tasks = []
        for target in target_agents:
            tasks.append(self.consult(source_agent, target, prompt, context, db))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        responses = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                responses.append({
                    "target": target_agents[i],
                    "status": "error",
                    "reason": str(result),
                })
            else:
                responses.append(result)

        successful = sum(1 for r in responses if r.get("status") == "success")
        logger.info(f"A2A: broadcast from {source_agent} → {successful}/{len(target_agents)} succeeded")
        return responses


# ─── Singleton ──────────────────────────────────────────────
_gateway: A2AGateway | None = None


def get_a2a_gateway() -> A2AGateway:
    global _gateway
    if _gateway is None:
        _gateway = A2AGateway()
    return _gateway
