"""
AI Agent Hub — Software Factory registry and orchestrator.

Config-driven: agents self-register from configuration.
New agents = new YAML config + one-line class. Zero hub changes.
"""

import logging

from app.services.agents.base import BaseAgent
from app.services.drivers.base import LlmResponse
from app.services.llm_service import get_llm_service

logger = logging.getLogger(__name__)


class AiAgentHub:
    """
    Central registry and orchestrator for all AI agents.

    Software Factory pattern:
    - Agents self-register via config-driven discovery
    - The hub routes tasks to the correct agent by identifier
    - Adding a new agent = add YAML config + class, register here
    """

    def __init__(self):
        self._agents: dict[str, BaseAgent] = {}
        self._auto_register()

    def _auto_register(self):
        """Auto-register all built-in agents. Software Factory: config-driven assembly."""
        from app.services.agents.copywriter import CopywriterAgent
        from app.services.agents.seo import SeoAgent
        from app.services.agents.social import SocialAgent
        from app.services.agents.email_campaign import EmailCampaignAgent
        from app.services.agents.whatsapp import WhatsappAgent
        from app.services.agents.sms import SmsAgent
        from app.services.agents.ad_creative import AdCreativeAgent

        llm = get_llm_service()
        for agent_cls in [CopywriterAgent, SeoAgent, SocialAgent, EmailCampaignAgent, WhatsappAgent, SmsAgent, AdCreativeAgent]:
            agent = agent_cls(llm)
            self.register(agent)

    def register(self, agent: BaseAgent) -> None:
        """Register an agent in the hub."""
        self._agents[agent.identifier] = agent
        logger.info(f"AiAgentHub: registered agent '{agent.identifier}'")

    def get(self, identifier: str) -> BaseAgent:
        """Resolve an agent by identifier."""
        agent = self._agents.get(identifier)
        if not agent:
            raise ValueError(f"Agent '{identifier}' not registered. Available: {list(self._agents.keys())}")
        return agent

    def available_agents(self) -> list[str]:
        """List all registered agent identifiers."""
        return list(self._agents.keys())

    def agent_info(self) -> list[dict]:
        """Get metadata for all registered agents."""
        return [agent.info() for agent in self._agents.values()]

    async def run(
        self, 
        agent_type: str, 
        prompt: str, 
        db: Any | None = None,
        context: dict | None = None, 
        **options
    ) -> LlmResponse:
        """Dispatch a task to the appropriate agent."""
        agent = self.get(agent_type)
        logger.info(f"AiAgentHub: running agent '{agent_type}'")
        return await agent.execute(prompt, db=db, context=context, **options)

    async def run_in_conversation(
        self,
        agent_type: str,
        prompt: str,
        history: list[dict],
        db: Any | None = None,
        context: dict | None = None,
        **options,
    ) -> LlmResponse:
        """Run a task within a conversation with full history."""
        agent = self.get(agent_type)
        return await agent.execute_in_conversation(prompt, history, db=db, context=context, **options)

    async def batch(
        self, 
        prompt: str, 
        agent_types: list[str], 
        db: Any | None = None,
        context: dict | None = None, 
        **options
    ) -> dict[str, LlmResponse]:
        """Run multiple agents in parallel on the same prompt."""
        import asyncio

        async def _run(agent_type: str):
            return agent_type, await self.run(agent_type, prompt, db=db, context=context, **options)

        results = await asyncio.gather(*[_run(t) for t in agent_types], return_exceptions=True)

        output = {}
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"AiAgentHub: batch agent failed: {result}")
            else:
                agent_type, response = result
                output[agent_type] = response

        return output


# ─── Singleton ──────────────────────────────────────────────────

_hub: AiAgentHub | None = None


def get_agent_hub() -> AiAgentHub:
    global _hub
    if _hub is None:
        _hub = AiAgentHub()
    return _hub
