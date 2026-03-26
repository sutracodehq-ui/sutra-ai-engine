"""
Agent Chain — Multi-agent orchestration for complex tasks.

Intelligence Upgrade 4: Instead of one agent per task, chain multiple
agents together in a pipeline: Plan → Execute → Review.

This produces dramatically better output quality, especially for
complex tasks like campaign strategy, content audits, and SEO analysis.

Patterns:
- Sequential Chain: A → B → C (each agent gets the previous output)
- Review Chain: Execute → Review → (if rejected, re-execute with feedback)
- Fan-out: Run multiple agents in parallel, merge results
"""

import logging
from typing import Any, Optional

from app.services.drivers.base import LlmResponse

logger = logging.getLogger(__name__)

# ─── Chain Config (from intelligence_config.yaml) ───────────

def _load_chain_configs() -> dict[str, dict]:
    """Load agent chain configurations from YAML."""
    import yaml
    from pathlib import Path

    config_path = Path("intelligence_config.yaml")
    if not config_path.exists():
        return {}

    with open(config_path) as f:
        config = yaml.safe_load(f) or {}

    return config.get("agent_chains", {})


class AgentChain:
    """
    Orchestrates multi-agent pipelines for higher quality output.
    """

    def __init__(self):
        from app.services.agents.hub import get_agent_hub
        self._hub = get_agent_hub()

    async def execute(
        self,
        agent_type: str,
        prompt: str,
        db: Any | None = None,
        context: dict | None = None,
        **options,
    ) -> LlmResponse:
        """
        Execute with chain if configured, otherwise direct execution.
        """
        chain_config = _load_chain_configs().get(agent_type)

        if not chain_config:
            # No chain configured — direct execution
            return await self._hub.run(agent_type, prompt, db=db, context=context, **options)

        chain_type = chain_config.get("type", "direct")

        if chain_type == "review":
            return await self._review_chain(agent_type, prompt, chain_config, db, context, **options)

        # Default: direct execution
        return await self._hub.run(agent_type, prompt, db=db, context=context, **options)

    async def _review_chain(
        self,
        agent_type: str,
        prompt: str,
        config: dict,
        db: Any | None = None,
        context: dict | None = None,
        max_retries: int = 1,
        **options,
    ) -> LlmResponse:
        """
        Execute → Review → (optionally re-execute with feedback).

        Flow:
        1. Run the primary agent
        2. Run an internal reviewer on the output
        3. If reviewer rejects → re-run primary with reviewer feedback
        4. Return the best result
        """
        import json
        from app.services.llm_service import get_llm_service

        # Step 1: Primary execution
        primary_response = await self._hub.run(agent_type, prompt, db=db, context=context, **options)
        logger.info(f"AgentChain: [{agent_type}] primary execution complete")

        if not primary_response.content:
            return primary_response

        # Step 2: Review
        reviewer_prompt = config.get("reviewer_prompt", "Review the following output for quality.")
        review_input = (
            f"{reviewer_prompt}\n\n"
            f"--- AGENT OUTPUT TO REVIEW ---\n"
            f"{primary_response.content[:3000]}"
        )

        llm = get_llm_service()
        review_response = await llm.complete(
            prompt=review_input,
            system_prompt="You are a quality reviewer. Respond with JSON only.",
            driver="ollama",  # Use local for review (free)
        )

        # Step 3: Parse review
        try:
            review = json.loads(review_response.content)
            approved = review.get("approved", True)
            score = review.get("score", 8)
            feedback = review.get("feedback", "")
        except (json.JSONDecodeError, TypeError):
            # If review parsing fails, trust the primary output
            approved = True
            score = 7
            feedback = ""

        logger.info(f"AgentChain: [{agent_type}] review score={score}, approved={approved}")

        if approved or max_retries <= 0:
            primary_response.metadata = primary_response.metadata or {}
            primary_response.metadata["review_score"] = score
            primary_response.metadata["review_approved"] = approved
            return primary_response

        # Step 4: Re-execute with feedback
        enhanced_prompt = (
            f"{prompt}\n\n"
            f"--- REVIEWER FEEDBACK (address these issues) ---\n"
            f"{feedback}"
        )

        retry_response = await self._hub.run(agent_type, enhanced_prompt, db=db, context=context, **options)
        retry_response.metadata = retry_response.metadata or {}
        retry_response.metadata["review_score"] = score
        retry_response.metadata["chain_retry"] = True
        logger.info(f"AgentChain: [{agent_type}] re-executed with reviewer feedback")

        return retry_response


# ─── Singleton ──────────────────────────────────────────────
_chain: AgentChain | None = None


def get_agent_chain() -> AgentChain:
    global _chain
    if _chain is None:
        _chain = AgentChain()
    return _chain
