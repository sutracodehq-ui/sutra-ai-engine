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
- Sequential: A -> B -> C pipeline
- Map-Reduce: Parallel execution + aggregation
- Review-Chain: Producer -> Reviewer -> Producer (loop)
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
    5. Execute complex pipelines (Sequential, Map-Reduce, Review-Chain)
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

    async def share_context(self, source_agent: str, target_agent: str, context: str):
        """Share context between agents via Memory."""
        from app.services.intelligence.memory import get_memory
        mem = get_memory()
        await mem.remember(target_agent, f"Shared from {source_agent}", context)

    async def handoff(
        self, source_agent: str, target_agent: str,
        prompt: str, context: dict | None = None,
        swarm_id: str = "", visited: set | None = None, db=None,
    ) -> dict:
        """
        Transfer execution control from one agent to another.

        Anti-loop guard: tracks visited agents per swarm session.
        If target was already visited, rejects the handoff.
        """
        visited = visited or set()

        # Circular delegation guard
        if target_agent in visited:
            logger.warning(f"A2A: CIRCULAR handoff rejected: {source_agent} → {target_agent} (visited={visited})")
            return {"status": "circular_rejected", "source": source_agent, "target": target_agent}

        visited.add(source_agent)
        visited.add(target_agent)

        # Permission check
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
                "a2a_type": "handoff",
                "swarm_id": swarm_id,
                "visited_agents": list(visited),
            }
            response = await agent.execute(prompt, db=db, context=a2a_context)
            logger.info(f"A2A: {source_agent} → handoff → {target_agent} ✓ (swarm={swarm_id})")
            return {
                "status": "success",
                "source": source_agent,
                "target": target_agent,
                "response": response.content,
                "visited": list(visited),
            }
        except Exception as e:
            logger.error(f"A2A: handoff to {target_agent} failed: {e}")
            return {"status": "error", "reason": str(e)}

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

    # ─── Advanced Pipelines ─────────────────────────────────

    async def pipeline_sequential(
        self,
        source_agent: str,
        agents: list[str],
        initial_prompt: str,
        db=None,
    ) -> dict:
        """
        Execute a sequential pipeline: Agent 1 -> Agent 2 -> Agent 3.
        Each agent receives the output of the previous agent as context.
        """
        current_input = initial_prompt
        history = []

        for i, agent_id in enumerate(agents):
            logger.info(f"A2A Sequential: [{i+1}/{len(agents)}] running {agent_id}")
            res = await self.consult(source_agent, agent_id, current_input, db=db)
            
            if res.get("status") != "success":
                return {"status": "pipeline_failed", "at_step": i, "agent": agent_id, "error": res.get("reason")}
            
            current_output = res.get("response", "")
            history.append({"agent": agent_id, "output": current_output})
            current_input = f"Context from previous agent ({agent_id}):\n{current_output}\n\nTask: Process this further."

        return {
            "status": "success",
            "final_output": current_input.split("Task:")[0].strip(), # Get last output
            "history": history
        }

    async def pipeline_map_reduce(
        self,
        source_agent: str,
        mapping_agents: list[str],
        reduction_agent: str,
        prompt: str,
        db=None,
    ) -> dict:
        """
        Parallel execution (Map) followed by a final aggregation (Reduce).
        Example: 3 Analysts (Map) -> 1 Strategist (Reduce).
        """
        import asyncio
        
        # 1. Map Phase (Parallel)
        logger.info(f"A2A Map-Reduce: Mapping to {len(mapping_agents)} agents")
        map_results = await self.broadcast(source_agent, mapping_agents, prompt, db=db)
        
        # Collect successful mappings
        insights = []
        for r in map_results:
            if r.get("status") == "success":
                insights.append(f"Insight from {r['target']}:\n{r.get('response')}")

        if not insights:
            return {"status": "map_failed", "reason": "No successful map results"}

        # 2. Reduce Phase
        logger.info(f"A2A Map-Reduce: Reducing via {reduction_agent}")
        reduce_prompt = f"Aggregate and synthesize the following insights into a unified strategy:\n\n" + "\n\n".join(insights)
        
        res = await self.consult(source_agent, reduction_agent, reduce_prompt, db=db)
        return res

    async def pipeline_review_chain(
        self,
        source_agent: str,
        producer_agent: str,
        reviewer_agent: str,
        prompt: str,
        max_iterations: int = 2,
        db=None,
    ) -> dict:
        """
        Iterative refinement: Producer generates -> Reviewer critiques -> Producer improves.
        """
        current_draft = ""
        feedback = ""
        
        for i in range(max_iterations):
            logger.info(f"A2A Review-Chain: Iteration [{i+1}/{max_iterations}]")
            
            # 1. Produce
            prod_prompt = prompt if i == 0 else f"Original Request: {prompt}\n\nPrevious Draft: {current_draft}\n\nReviewer Feedback: {feedback}\n\nPlease improve the draft based on the feedback."
            prod_res = await self.consult(source_agent, producer_agent, prod_prompt, db=db)
            if prod_res.get("status") != "success":
                return prod_res
            
            current_draft = prod_res.get("response", "")
            
            # 2. Review
            rev_prompt = f"Review the following output from {producer_agent} for accuracy, tone, and compliance:\n\n{current_draft}\n\nProvide constructive feedback or say 'APPROVED' if it is perfect."
            rev_res = await self.consult(source_agent, reviewer_agent, rev_prompt, db=db)
            if rev_res.get("status") != "success":
                return rev_res
            
            feedback = rev_res.get("response", "")
            if "APPROVED" in feedback.upper():
                logger.info(f"A2A Review-Chain: Approved after {i+1} iterations.")
                break

        return {
            "status": "success",
            "final_draft": current_draft,
            "reviews_done": i + 1,
            "last_feedback": feedback
        }

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

    # ─── Teach (push insight to peer) ───────────────────────

    async def teach(
        self,
        source_agent: str,
        target_agent: str,
        insight: str,
        topic: str = "general",
    ) -> dict:
        """
        One agent teaches another by pushing a learned insight.
        Stores the insight in the target agent's memory.
        """
        try:
            from app.services.intelligence.memory import get_memory
            mem = get_memory()
            await mem.remember(
                agent_type=target_agent,
                prompt=f"Teaching Session: {topic}",
                response=f"Insight from {source_agent}: {insight}",
                quality_score=0.9
            )

            logger.info(f"A2A: {source_agent} → teach → {target_agent} (success)")
            return {
                "status": "success",
                "source": source_agent,
                "target": target_agent,
                "topic": topic,
            }
        except Exception as e:
            logger.error(f"A2A: teach failed: {e}")
            return {"status": "error", "reason": str(e)}

    # ─── Request Learning (pull knowledge from peers) ───────

    async def request_learning(
        self,
        agent_id: str,
        topic: str,
        db=None,
    ) -> dict:
        """
        An agent requests knowledge from its alliance peers on a topic.

        Discovers the agent's alliance, broadcasts the topic to peers,
        and collects relevant insights.
        """
        routes = _load_a2a_routes()

        # Find the agent's alliance from cross_teaching config
        alliance_members = self._get_alliance_members(agent_id)

        if not alliance_members:
            return {
                "status": "no_alliance",
                "reason": f"{agent_id} has no teaching alliance configured",
            }

        # Filter to only peers (not self)
        peers = [m for m in alliance_members if m != agent_id]
        if not peers:
            return {"status": "no_peers"}

        # Broadcast the topic to peers
        prompt = (
            f"Share your expertise on: {topic}. "
            f"Provide one concise, actionable insight that would help a peer agent."
        )
        responses = await self.broadcast(agent_id, peers, prompt, db=db)

        # Collect successful insights
        insights = []
        for r in responses:
            if r.get("status") == "success" and r.get("response"):
                insights.append({
                    "from": r["target"],
                    "insight": r["response"][:500],
                })

        logger.info(
            f"A2A: {agent_id} requested learning on '{topic}' "
            f"→ {len(insights)}/{len(peers)} peers responded"
        )
        return {
            "status": "success",
            "agent": agent_id,
            "topic": topic,
            "insights_received": len(insights),
            "insights": insights,
        }

    def _get_alliance_members(self, agent_id: str) -> list[str]:
        """Find the alliance members for a given agent."""
        if not CONFIG_PATH.exists():
            return []
        with open(CONFIG_PATH) as f:
            config = yaml.safe_load(f) or {}
        alliances = config.get("cross_teaching", {}).get("alliances", {})

        for _alliance_name, alliance_config in alliances.items():
            members = alliance_config.get("members", [])
            if agent_id in members:
                return members
        return []


# ─── Singleton ──────────────────────────────────────────────
_gateway: A2AGateway | None = None


def get_a2a_gateway() -> A2AGateway:
    global _gateway
    if _gateway is None:
        _gateway = A2AGateway()
    return _gateway

