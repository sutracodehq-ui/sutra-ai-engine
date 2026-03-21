"""
Agent Orchestrator — Multi-agent task decomposition and synthesis.

Software Factory Principle: Automation + Modular Architecture.

Takes a complex query, decomposes it into sub-tasks, routes each to
the best specialist agent, executes them (parallel where possible),
and synthesizes results into a unified response.

Architecture:
    Complex Query → Decompose → Route → Execute (parallel) → Synthesize
"""

import asyncio
import json
import logging
from typing import Optional

import yaml
from pathlib import Path

from app.config import get_settings

logger = logging.getLogger(__name__)


def _load_orchestrator_config() -> dict:
    """Load orchestrator config from YAML."""
    config_path = Path("intelligence_config.yaml")
    if not config_path.exists():
        return {}
    with open(config_path) as f:
        config = yaml.safe_load(f) or {}
    return config.get("orchestrator", {})


class AgentOrchestrator:
    """
    Multi-agent orchestration engine.

    Decomposes complex queries into sub-tasks and coordinates
    multiple specialized agents to produce a synthesized answer.
    """

    def __init__(self):
        self._hub = None
        self._a2a = None

    def _get_hub(self):
        if self._hub is None:
            from app.services.agents.hub import AiAgentHub
            self._hub = AiAgentHub()
        return self._hub

    def _get_a2a(self):
        if self._a2a is None:
            from app.services.intelligence.a2a_gateway import get_a2a_gateway
            self._a2a = get_a2a_gateway()
        return self._a2a

    # ─── Task Decomposition ─────────────────────────────────

    async def decompose(self, query: str, db=None) -> list[dict]:
        """
        Use the LLM to break a complex query into sub-tasks,
        each tagged with the best agent to handle it.
        """
        config = _load_orchestrator_config()
        hub = self._get_hub()

        # Get available agents for the decomposer prompt
        available = []
        for identifier in hub._registry:
            agent = hub.get(identifier)
            if agent:
                info = agent.info()
                available.append(f"- {identifier}: {info.get('description', '')[:100]}")

        agents_list = "\n".join(available)

        decompose_prompt = f"""You are a task decomposition engine. Break this complex query into sub-tasks.

Available specialist agents:
{agents_list}

For each sub-task, specify:
- "task": what the agent should do (specific prompt)
- "agent": which agent should handle it (from the list above)
- "depends_on": list of task indices this depends on (empty if independent)

Return ONLY valid JSON array. Example:
[
  {{"task": "Analyze stock technicals for NVDA", "agent": "stock_analyzer", "depends_on": []}},
  {{"task": "Check AI industry trends affecting NVDA", "agent": "ai_trend_tracker", "depends_on": []}},
  {{"task": "Synthesize findings into recommendation", "agent": "stock_predictor", "depends_on": [0, 1]}}
]

Query: {query}"""

        from app.services.llm_service import get_llm_service
        llm = get_llm_service()
        response = await llm.complete(
            prompt=decompose_prompt,
            system_prompt="You are a precise task decomposition engine. Return only valid JSON.",
        )

        try:
            # Extract JSON from response
            import re
            json_match = re.search(r'\[.*\]', response.content, re.DOTALL)
            if json_match:
                tasks = json.loads(json_match.group())
                logger.info(f"Orchestrator: decomposed into {len(tasks)} sub-tasks")
                return tasks
        except (json.JSONDecodeError, AttributeError) as e:
            logger.warning(f"Orchestrator: decomposition parse failed: {e}")

        # Fallback: single-task execution
        return [{"task": query, "agent": "copywriter", "depends_on": []}]

    # ─── Execution ──────────────────────────────────────────

    async def execute(self, query: str, db=None, context: dict | None = None) -> dict:
        """
        Full orchestration cycle:
        1. Decompose query into sub-tasks
        2. Execute independent tasks in parallel
        3. Execute dependent tasks sequentially
        4. Synthesize all results
        """
        # 1. Decompose
        tasks = await self.decompose(query, db)

        # 2. Execute in dependency order
        results = {}
        hub = self._get_hub()

        # Group tasks by dependency level
        levels = self._topological_sort(tasks)

        for level in levels:
            level_coroutines = []
            level_indices = []

            for idx in level:
                task = tasks[idx]
                agent_id = task.get("agent", "copywriter")
                task_prompt = task["task"]

                # Inject results from dependencies into prompt
                deps = task.get("depends_on", [])
                if deps:
                    dep_context = "\n\n".join(
                        f"[Result from {tasks[d]['agent']}]: {results.get(d, 'N/A')}"
                        for d in deps if d in results
                    )
                    task_prompt = f"{task_prompt}\n\nContext from previous analyses:\n{dep_context}"

                agent = hub.get(agent_id)
                if agent:
                    level_coroutines.append(
                        agent.execute(task_prompt, db=db, context=context)
                    )
                    level_indices.append(idx)

            # Execute this level in parallel
            if level_coroutines:
                level_results = await asyncio.gather(
                    *level_coroutines, return_exceptions=True
                )
                for i, result in enumerate(level_results):
                    idx = level_indices[i]
                    if isinstance(result, Exception):
                        results[idx] = f"Error: {result}"
                    else:
                        results[idx] = result.content

        # 3. Synthesize
        synthesis = await self._synthesize(query, tasks, results, db)

        return {
            "query": query,
            "sub_tasks": len(tasks),
            "tasks": [
                {
                    "agent": t["agent"],
                    "task": t["task"][:100],
                    "result_preview": str(results.get(i, ""))[:200],
                }
                for i, t in enumerate(tasks)
            ],
            "synthesis": synthesis,
        }

    # ─── Synthesis ──────────────────────────────────────────

    async def _synthesize(
        self, original_query: str, tasks: list[dict], results: dict, db=None
    ) -> str:
        """Merge results from all sub-tasks into a unified response."""
        results_text = "\n\n".join(
            f"### {tasks[i]['agent']} result:\n{content}"
            for i, content in results.items()
        )

        synthesis_prompt = f"""You are a synthesis expert. Combine these specialist analyses into one unified, coherent response.

Original question: {original_query}

Specialist results:
{results_text}

Create a comprehensive, well-structured response that:
1. Integrates all specialist insights
2. Resolves any contradictions
3. Provides a clear overall conclusion
4. Is actionable and specific"""

        from app.services.llm_service import get_llm_service
        llm = get_llm_service()
        response = await llm.complete(
            prompt=synthesis_prompt,
            system_prompt="You synthesize multi-agent analyses into clear, actionable insights.",
        )
        return response.content

    # ─── Topological Sort ───────────────────────────────────

    def _topological_sort(self, tasks: list[dict]) -> list[list[int]]:
        """Sort tasks into execution levels based on dependencies."""
        n = len(tasks)
        in_degree = [0] * n
        adj = [[] for _ in range(n)]

        for i, task in enumerate(tasks):
            for dep in task.get("depends_on", []):
                if 0 <= dep < n:
                    adj[dep].append(i)
                    in_degree[i] += 1

        # BFS by levels
        levels = []
        queue = [i for i in range(n) if in_degree[i] == 0]

        while queue:
            levels.append(queue[:])
            next_queue = []
            for node in queue:
                for neighbor in adj[node]:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        next_queue.append(neighbor)
            queue = next_queue

        # Add any remaining nodes (circular deps)
        placed = {i for level in levels for i in level}
        remaining = [i for i in range(n) if i not in placed]
        if remaining:
            levels.append(remaining)

        return levels


# ─── Singleton ──────────────────────────────────────────────
_orchestrator: AgentOrchestrator | None = None


def get_agent_orchestrator() -> AgentOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = AgentOrchestrator()
    return _orchestrator
