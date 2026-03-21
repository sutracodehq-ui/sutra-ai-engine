"""
Meta-Prompt Optimizer — auto-evolves agent system prompts.

Self-Learning Tier 3: Analyzes failed/low-rated responses, uses a
meta-LLM (Groq, free & fast) to generate improved system prompts,
and stores them as A/B test candidates via AgentOptimization.

Flow:
1. Query AgentFeedback for low-quality signals (rejected/regenerated)
2. Group by agent_type
3. For each agent: send failures to Groq meta-prompt → get improved system prompt
4. Store as AgentOptimization candidate (is_active=False)
5. BaseAgent's existing A/B system (10% explore) auto-tests it
6. If candidate beats active prompt after N trials → auto-promote
"""

import logging
from typing import Optional

from sqlalchemy import select, func, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.agent_feedback import AgentFeedback
from app.models.agent_optimization import AgentOptimization
from app.services.llm_service import get_llm_service

logger = logging.getLogger(__name__)

# Meta-prompt template for generating improved system prompts
META_PROMPT = """You are an expert prompt engineer. Analyze the following failed AI agent interactions 
and generate an IMPROVED system prompt for the agent.

## Agent Type: {agent_type}
## Current System Prompt:
{current_prompt}

## Failed Interactions (user rejected or regenerated):
{failures}

## Requirements:
1. Keep the same domain expertise and capabilities
2. Fix the patterns that caused failures
3. Be more specific about output format (JSON structure, required fields)
4. Add guardrails against common failure modes
5. Keep the prompt concise — no longer than 500 words

## Output:
Return ONLY the improved system prompt text. No explanations, no markdown fences."""


class MetaPromptOptimizer:
    """
    Analyzes agent failures and generates improved system prompts.
    """

    def __init__(self, db: AsyncSession):
        self._db = db

    async def get_failure_summary(self, agent_type: str, limit: int = 10) -> list[dict]:
        """Get recent failures for an agent type."""
        stmt = (
            select(AgentFeedback)
            .where(
                and_(
                    AgentFeedback.agent_type == agent_type,
                    AgentFeedback.quality_score < 0.0,
                )
            )
            .order_by(AgentFeedback.created_at.desc())
            .limit(limit)
        )
        result = await self._db.execute(stmt)
        feedbacks = result.scalars().all()

        failures = []
        for fb in feedbacks:
            task = fb.task
            failures.append({
                "signal": fb.signal,
                "comment": fb.comment,
                "prompt": getattr(task, "prompt", "N/A") if task else "N/A",
                "response_excerpt": (getattr(task, "response", "") or "")[:500] if task else "N/A",
            })
        return failures

    async def get_agents_needing_optimization(self, min_failures: int = 3) -> list[str]:
        """Find agent types with enough failures to warrant optimization."""
        stmt = (
            select(AgentFeedback.agent_type, func.count(AgentFeedback.id).label("fail_count"))
            .where(AgentFeedback.quality_score < 0.0)
            .group_by(AgentFeedback.agent_type)
            .having(func.count(AgentFeedback.id) >= min_failures)
        )
        result = await self._db.execute(stmt)
        return [row[0] for row in result.all()]

    async def optimize(self, agent_type: str) -> Optional[AgentOptimization]:
        """
        Generate an improved system prompt for an agent type.

        Uses Groq (free, fast) as the meta-LLM to analyze failures
        and generate a better system prompt.
        """
        # 1. Get current system prompt from YAML
        current_prompt = self._load_current_prompt(agent_type)
        if not current_prompt:
            logger.warning(f"MetaOptimizer: no YAML config for {agent_type}, skipping")
            return None

        # 2. Get failure summary
        failures = await self.get_failure_summary(agent_type)
        if not failures:
            logger.info(f"MetaOptimizer: no failures for {agent_type}, skipping")
            return None

        # 3. Build meta-prompt
        failures_text = "\n".join(
            f"- Signal: {f['signal']}, Comment: {f['comment'] or 'none'}, "
            f"Prompt: {f['prompt'][:200]}, Response: {f['response_excerpt'][:200]}"
            for f in failures
        )

        meta_input = META_PROMPT.format(
            agent_type=agent_type,
            current_prompt=current_prompt,
            failures=failures_text,
        )

        # 4. Call Groq (free, fast) to generate improved prompt
        llm = get_llm_service()
        response = await llm.complete(
            prompt=meta_input,
            system_prompt="You are a world-class prompt engineer.",
            driver="groq",  # Always use Groq for meta-optimization (free)
        )

        if not response.content:
            logger.warning(f"MetaOptimizer: empty response for {agent_type}")
            return None

        # 5. Get next version number
        stmt = (
            select(func.max(AgentOptimization.version))
            .where(AgentOptimization.agent_type == agent_type)
        )
        result = await self._db.execute(stmt)
        max_version = result.scalar() or 0

        # 6. Store as candidate (NOT active — A/B tested via explore rate)
        optimization = AgentOptimization(
            agent_type=agent_type,
            version=max_version + 1,
            prompt_text=response.content.strip(),
            notes=f"Auto-generated from {len(failures)} failures. Model: {response.model}",
            is_active=False,
        )
        self._db.add(optimization)
        await self._db.commit()
        await self._db.refresh(optimization)

        logger.info(
            f"MetaOptimizer: created v{optimization.version} for {agent_type} "
            f"(analyzed {len(failures)} failures)"
        )
        return optimization

    def _load_current_prompt(self, agent_type: str) -> Optional[str]:
        """Load the current system prompt from YAML config."""
        import yaml
        from pathlib import Path

        config_path = Path("agent_config") / f"{agent_type}.yaml"
        if not config_path.exists():
            return None

        with open(config_path) as f:
            config = yaml.safe_load(f)

        return config.get("system_prompt", config.get("domain", ""))

    async def run_optimization_cycle(self) -> dict:
        """
        Full optimization cycle: find agents needing improvement → optimize each.

        Returns: {optimized: [agent_type, ...], skipped: [...], errors: [...]}
        """
        result = {"optimized": [], "skipped": [], "errors": []}

        agents = await self.get_agents_needing_optimization()
        if not agents:
            logger.info("MetaOptimizer: no agents need optimization this cycle")
            return result

        for agent_type in agents:
            try:
                opt = await self.optimize(agent_type)
                if opt:
                    result["optimized"].append(agent_type)
                else:
                    result["skipped"].append(agent_type)
            except Exception as e:
                logger.error(f"MetaOptimizer: error optimizing {agent_type}: {e}")
                result["errors"].append({"agent_type": agent_type, "error": str(e)})

        return result
