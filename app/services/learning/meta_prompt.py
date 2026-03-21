"""
Meta-Prompt Optimizer — uses OPRO (Optimization by PROmpting) logic.

Analyzes high-quality and low-quality samples (based on user feedback)
to automatically refine and evolve agent system prompts.
"""

import logging
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.ai_task import AiTask
from app.models.agent_feedback import AgentFeedback
from app.services.llm_service import get_llm_service
from app.config import get_settings

logger = logging.getLogger(__name__)


class MetaPromptService:
    """Service for autonomous prompt optimization."""

    @classmethod
    async def optimize_agent(cls, db: AsyncSession, agent_type: str) -> str | None:
        """
        Analyze recent feedback for an agent and suggest an improved system prompt.
        """
        # 1. Fetch samples with feedback
        samples = await cls._get_feedback_samples(db, agent_type)
        if not samples:
            logger.info(f"Learn Loop: Not enough feedback samples for '{agent_type}'")
            return None

        # 2. Prepare Meta-Prompt
        meta_prompt = cls._build_meta_prompt(agent_type, samples)

        # 3. Call Meta-LLM to generate a better version
        settings = get_settings()
        service = get_llm_service()
        
        logger.info(f"Learn Loop: Optimizing '{agent_type}' using {settings.ai_meta_prompt_model}")
        
        result = await service.complete(
            prompt=meta_prompt,
            system_prompt=(
                "You are an expert Prompt Engineer. Your goal is to analyze user feedback "
                "on AI responses and improve the system prompt to maximize quality scores."
            ),
            model=settings.ai_meta_prompt_model,
            temperature=0.2  # Low temperature for stable optimization
        )

        new_prompt = result.get("content")
        return new_prompt

    @staticmethod
    async def _get_feedback_samples(db: AsyncSession, agent_type: str, limit: int = 50) -> list[dict]:
        """Fetch tasks with their associated feedback."""
        stmt = (
            select(AiTask)
            .join(AgentFeedback)
            .where(AiTask.agent_type == agent_type)
            .options(selectinload(AiTask.feedback))
            .order_by(AiTask.created_at.desc())
            .limit(limit)
        )
        result = await db.execute(stmt)
        tasks = result.scalars().all()

        samples = []
        for task in tasks:
            if not task.feedback:
                continue
            
            samples.append({
                "prompt": task.prompt,
                "response": task.result.get("content") if task.result else "",
                "score": task.feedback.quality_score,
                "signal": task.feedback.signal,
                "user_comment": task.feedback.comment
            })
        
        return samples

    @staticmethod
    def _build_meta_prompt(agent_type: str, samples: list[dict]) -> str:
        """Construct the prompt for the Meta-Optimizer."""
        
        # Sort samples by score: bad first, then good
        sorted_samples = sorted(samples, key=lambda x: x["score"])
        
        samples_text = ""
        for i, s in enumerate(sorted_samples):
            status = "✅ SUCCESS" if s["score"] > 0 else "❌ FAILURE"
            samples_text += f"\n--- Sample {i+1} ({status}, Signal: {s['signal']}) ---\n"
            samples_text += f"USER PROMPT: {s['prompt'][:200]}...\n"
            samples_text += f"AI RESPONSE: {s['response'][:300]}...\n"
            if s["user_comment"]:
                samples_text += f"USER FEEDBACK: {s['user_comment']}\n"

        return f"""
I need you to optimize the system prompt for an AI agent called '{agent_type}'.
Below are several real-world samples of how the current agent has performed, including user feedback and success/failure signals.

{samples_text}

### Instructions:
1. Analyze the patterns in the failures. Why did the user reject or regenerate those responses?
2. Analyze the successes. What works well?
3. Rewrite the SYSTEM PROMPT for the '{agent_type}' agent to avoid the observed failures while maintaining the successes.
4. Your output must ONLY be the new system prompt text. Do not include explanations.

NEW SYSTEM PROMPT:
""".strip()
