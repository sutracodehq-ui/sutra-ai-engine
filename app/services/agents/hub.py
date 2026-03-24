"""
AI Agent Hub — Software Factory registry and orchestrator.

Config-driven: agents auto-register from YAML configs in agent_config/.
New agents = new YAML file. Zero code changes needed.
"""

import logging
import threading
from pathlib import Path
from typing import Any, AsyncGenerator

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
        """
        Auto-discover and register all agents from YAML configs.

        Software Factory: instead of 187 manual imports, scan agent_config/
        and create a SutraAgent for each YAML file found. Adding a new skill
        = just add a YAML file. Zero code changes needed.
        """
        from app.services.agents.sutra_agent import SutraAgent
        from app.services.llm_service import get_llm_service

        llm = get_llm_service()
        config_dir = Path("agent_config")

        if not config_dir.exists():
            logger.warning("agent_config/ directory not found — no agents registered")
            return

        registered = 0
        for yaml_file in sorted(config_dir.glob("*.yaml")):
            skill_id = yaml_file.stem  # e.g., "quiz_generator" from "quiz_generator.yaml"
            try:
                agent = SutraAgent(skill_id, llm=llm)
                self.register(agent)
                registered += 1
            except Exception as e:
                logger.warning(f"Failed to register agent '{skill_id}': {e}")

        logger.info(f"AiAgentHub: auto-registered {registered} agents from {config_dir}")


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

    # ─── Intent-Based Smart Routing (Centralized) ────────────
    # Config-driven: keyword + action word → specialist agent.
    # Works for ALL execution paths: run(), run_stream(), run_in_conversation().
    # When user says "generate a quiz on X", the hub auto-swaps to quiz_generator
    # regardless of which agent was originally requested.

    INTENT_ROUTES: list[dict] = [
        # EdTech
        {"keywords": ["quiz", "mcq", "question paper", "test paper", "question bank"],
         "agent": "quiz_generator", "actions": ["generate", "create", "make", "build", "prepare"]},
        {"keywords": ["notes", "revision notes", "summary notes", "study notes", "study material"],
         "agent": "note_generator", "actions": ["generate", "create", "make", "write", "prepare"]},
        {"keywords": ["flashcard", "flash card"],
         "agent": "flashcard_creator", "actions": ["generate", "create", "make", "build"]},
        {"keywords": ["lecture plan", "lesson plan", "teaching plan", "class plan"],
         "agent": "lecture_planner", "actions": ["generate", "create", "make", "plan", "design"]},
        {"keywords": ["key points", "important points", "main points"],
         "agent": "key_points_extractor", "actions": ["extract", "list", "give", "find", "get"]},
        # Marketing
        {"keywords": ["social media post", "instagram post", "twitter post", "facebook post", "linkedin post"],
         "agent": "social", "actions": ["generate", "create", "write", "make", "draft"]},
        {"keywords": ["email campaign", "newsletter", "marketing email"],
         "agent": "email_campaign", "actions": ["generate", "create", "write", "draft"]},
        {"keywords": ["ad copy", "advertisement", "ad creative"],
         "agent": "ad_creative", "actions": ["generate", "create", "write", "make", "design"]},
        {"keywords": ["seo", "meta title", "meta description"],
         "agent": "seo", "actions": ["analyze", "generate", "optimize", "create", "write"]},
        # Finance
        {"keywords": ["stock", "share price", "equity"],
         "agent": "stock_analyzer", "actions": ["analyze", "check", "review"]},
        # Health
        {"keywords": ["diet plan", "meal plan", "nutrition plan"],
         "agent": "diet_planner", "actions": ["generate", "create", "make", "plan", "suggest"]},
        {"keywords": ["symptoms", "feeling sick", "health issue"],
         "agent": "symptom_triage", "actions": ["check", "assess", "evaluate", "help"]},
        # Legal
        {"keywords": ["contract", "agreement", "legal document"],
         "agent": "contract_analyzer", "actions": ["analyze", "review", "check", "draft"]},
        {"keywords": ["rti", "right to information"],
         "agent": "rti_drafter", "actions": ["draft", "write", "create", "file"]},
    ]

    def _resolve_agent(self, requested_agent: str, prompt: str) -> str:
        """
        Smart agent resolution — detect user intent and route to specialist.

        Priority:
        1. If prompt matches a specialist intent (keyword + action), use that specialist
        2. Otherwise, use the originally requested agent

        This ensures "generate a quiz on X" always goes to quiz_generator,
        even if the frontend called education_guru or chatbot_trainer.
        """
        # Skip intent routing if already targeting a specialist directly
        # (prevents re-routing quiz_generator → quiz_generator)
        specialist_ids = {r["agent"] for r in self.INTENT_ROUTES}
        if requested_agent in specialist_ids:
            return requested_agent

        # Skip if in a delegation chain (prevent loops)
        # This is checked via context in run() but we do a fast check here
        msg_lower = prompt.lower()

        for route in self.INTENT_ROUTES:
            # Only route to agents that exist
            if route["agent"] not in self._agents:
                continue

            # Keyword match + action word match
            has_keyword = any(kw in msg_lower for kw in route["keywords"])
            has_action = any(aw in msg_lower for aw in route["actions"])

            if has_keyword and has_action:
                logger.info(
                    f"AiAgentHub: intent routing {requested_agent} → {route['agent']} "
                    f"for: {prompt[:80]}"
                )
                return route["agent"]

        return requested_agent

    async def run(
        self, 
        agent_type: str, 
        prompt: str, 
        db: Any | None = None,
        context: dict | None = None, 
        **options
    ) -> LlmResponse:
        """Dispatch a task to the appropriate agent, with auto-delegation."""
        # Smart routing: detect specialist intent before execution
        resolved_type = self._resolve_agent(agent_type, prompt)
        agent = self.get(resolved_type)
        logger.info(f"AiAgentHub: running agent '{resolved_type}'")
        response = await agent.execute(prompt, db=db, context=context, **options)

        # ─── Auto-Delegation: detect delegate_to in response ─────
        # Skip if already in a delegation chain (prevent infinite loops)
        if context and context.get("_delegation_chain"):
            return response

        delegate_target = self._extract_delegation(response.content)
        if delegate_target and delegate_target != agent_type and delegate_target in self._agents:
            logger.info(f"AiAgentHub: auto-delegating from {agent_type} → {delegate_target}")
            delegation_result = await self.delegate(
                from_agent=agent_type,
                to_agent=delegate_target,
                prompt=prompt,
                context=context,
                db=db,
            )
            if delegation_result.get("status") == "success":
                # Merge: keep original response metadata but enrich content
                specialist = delegation_result["response"]
                response.content = (
                    f'{response.content}\n\n'
                    f'--- Specialist Insight from {delegate_target} ---\n'
                    f'{specialist}'
                )
                response.metadata = response.metadata or {}
                response.metadata["delegated_to"] = delegate_target
                response.metadata["delegation_chain"] = delegation_result.get("chain", [])

        return response

    @staticmethod
    def _extract_delegation(content: str) -> str | None:
        """Extract delegate_to agent identifier from a JSON response."""
        import json
        try:
            # Strip markdown code fences if present
            clean = content.strip()
            if clean.startswith("```"):
                lines = clean.split("\n")
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                clean = "\n".join(lines).strip()

            parsed = json.loads(clean)
            if isinstance(parsed, dict):
                return parsed.get("delegate_to")
        except (json.JSONDecodeError, ValueError):
            pass
        return None

    async def run_stream(
        self,
        agent_type: str,
        prompt: str,
        db: Any | None = None,
        context: dict | None = None,
        **options
    ) -> AsyncGenerator[str, None]:
        """Stream a task response token-by-token (SSE-ready)."""
        # Smart routing: detect specialist intent before streaming
        resolved_type = self._resolve_agent(agent_type, prompt)
        agent = self.get(resolved_type)
        logger.info(f"AiAgentHub: streaming agent '{resolved_type}'")
        async for token in agent.execute_stream(prompt, db=db, context=context, **options):
            yield token

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
        # Smart routing: detect specialist intent before conversation execution
        resolved_type = self._resolve_agent(agent_type, prompt)
        agent = self.get(resolved_type)
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

    # ─── Safe Inter-Agent Delegation ──────────────────────────

    MAX_DELEGATION_DEPTH = 3     # Max hops: A → B → C → D (stops)
    DELEGATION_TIMEOUT = 30      # Seconds before a delegation times out

    async def delegate(
        self,
        from_agent: str,
        to_agent: str,
        prompt: str,
        context: dict | None = None,
        db: Any | None = None,
        _chain: list[str] | None = None,
        _depth: int = 0,
    ) -> dict:
        """
        Safe inter-agent delegation with:
        1. Max depth (3 hops) — prevents infinite delegation chains
        2. Cycle detection — agent A→B→A stops immediately
        3. Timeout (30s) — agent never waits forever
        4. Fallback — if delegation fails, returns best-effort response

        Usage (from within any agent):
            hub = get_agent_hub()
            result = await hub.delegate(
                from_agent="trip_planner",
                to_agent="visa_guide",
                prompt="What's the visa process for Thailand?",
            )
        """
        import asyncio

        chain = _chain or [from_agent]

        # ── Guard: max depth ──
        if _depth >= self.MAX_DELEGATION_DEPTH:
            logger.warning(
                f"Delegation depth limit ({self.MAX_DELEGATION_DEPTH}) reached: "
                f"{'→'.join(chain)}→{to_agent}. Returning fallback."
            )
            return {
                "status": "fallback",
                "reason": "max_depth_reached",
                "chain": chain,
                "response": f"I consulted with {to_agent} but couldn't get a detailed answer in time. "
                            f"Here's what I know based on my own expertise.",
            }

        # ── Guard: cycle detection ──
        if to_agent in chain:
            logger.warning(
                f"Delegation cycle detected: {'→'.join(chain)}→{to_agent}. Breaking cycle."
            )
            return {
                "status": "fallback",
                "reason": "cycle_detected",
                "chain": chain,
                "response": f"I've already consulted with {to_agent} in this chain. "
                            f"Proceeding with available information.",
            }

        # ── Guard: agent exists ──
        if to_agent not in self._agents:
            return {
                "status": "fallback",
                "reason": "agent_not_found",
                "chain": chain,
                "response": f"The specialist '{to_agent}' is not available right now.",
            }

        # ── Execute with timeout ──
        chain_ext = chain + [to_agent]
        logger.info(f"Delegation: {'→'.join(chain_ext)} (depth={_depth + 1})")

        try:
            result = await asyncio.wait_for(
                self.run(to_agent, prompt, db=db, context={
                    **(context or {}),
                    "_delegation_chain": chain_ext,
                    "_delegation_depth": _depth + 1,
                }),
                timeout=self.DELEGATION_TIMEOUT,
            )
            return {
                "status": "success",
                "chain": chain_ext,
                "response": result.content if hasattr(result, "content") else str(result),
                "agent": to_agent,
            }
        except asyncio.TimeoutError:
            logger.warning(f"Delegation to '{to_agent}' timed out after {self.DELEGATION_TIMEOUT}s")
            return {
                "status": "fallback",
                "reason": "timeout",
                "chain": chain_ext,
                "response": f"The {to_agent} specialist is taking too long. "
                            f"I'll provide my best answer based on what I know.",
            }
        except Exception as e:
            logger.error(f"Delegation to '{to_agent}' failed: {e}")
            return {
                "status": "fallback",
                "reason": "error",
                "chain": chain_ext,
                "response": f"Couldn't reach the {to_agent} specialist. "
                            f"Providing my best answer instead.",
            }

    async def multi_delegate(
        self,
        from_agent: str,
        to_agents: list[str],
        prompt: str,
        context: dict | None = None,
        db: Any | None = None,
    ) -> dict[str, dict]:
        """
        Delegate to multiple agents in parallel.
        All run concurrently — if one is slow/fails, others still return.

        Example: trip_planner delegates to [visa_guide, cultural_advisor, travel_budget_optimizer]
        """
        import asyncio

        async def _delegate_one(target: str):
            return target, await self.delegate(
                from_agent=from_agent,
                to_agent=target,
                prompt=prompt,
                context=context,
                db=db,
            )

        results = await asyncio.gather(
            *[_delegate_one(t) for t in to_agents],
            return_exceptions=True,
        )

        output = {}
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Multi-delegation failed: {result}")
            else:
                agent_id, response = result
                output[agent_id] = response

        return output


# ─── Singleton ──────────────────────────────────────────────────

_hub: AiAgentHub | None = None
_hub_lock = threading.Lock()


def get_agent_hub() -> AiAgentHub:
    global _hub
    if _hub is None:
        with _hub_lock:
            if _hub is None:
                _hub = AiAgentHub()
    return _hub
