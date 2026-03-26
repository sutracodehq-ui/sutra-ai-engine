"""
Training Data Collector — converts feedback signals into fine-tuning data.

Self-Learning Tier 2: Watches AgentFeedback for high-quality signals
and exports them as JSONL training data for LoRA fine-tuning.

Signal mapping:
- accepted (1.0) → direct training example (prompt → response as-is)
- edited (0.5) → use the user's edited version as gold response
- rejected (-1.0) → excluded
- regenerated (-0.5) → excluded
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.agent_feedback import AgentFeedback

logger = logging.getLogger(__name__)

# Minimum quality score to include in training data
MIN_QUALITY_SCORE = 0.5
EXPORT_DIR = Path("training/data")


class TrainingDataCollector:
    """
    Collects feedback-rated agent interactions and exports them
    as JSONL for LoRA fine-tuning.
    """

    def __init__(self, db: AsyncSession):
        self._db = db

    async def collect(
        self,
        since: Optional[datetime] = None,
        agent_type: Optional[str] = None,
        min_score: float = MIN_QUALITY_SCORE,
    ) -> list[dict]:
        """
        Query feedback entries that meet quality threshold.

        Returns list of training examples in ChatML format.
        """
        conditions = [AgentFeedback.quality_score >= min_score]

        if since:
            conditions.append(AgentFeedback.created_at >= since)
        if agent_type:
            conditions.append(AgentFeedback.agent_type == agent_type)

        stmt = select(AgentFeedback).where(and_(*conditions)).order_by(AgentFeedback.created_at.desc())
        result = await self._db.execute(stmt)
        feedbacks = result.scalars().all()

        examples = []
        for fb in feedbacks:
            example = self._feedback_to_example(fb)
            if example:
                examples.append(example)

        logger.info(f"TrainingDataCollector: collected {len(examples)} examples from {len(feedbacks)} feedbacks")
        return examples

    def _feedback_to_example(self, fb: AgentFeedback) -> Optional[dict]:
        """
        Convert a single feedback entry to a ChatML training example.

        For 'edited' signals, uses the user's edited version as the gold response.
        For 'accepted' signals, uses the original AI response.
        """
        try:
            task = fb.task
            if not task:
                return None

            # Get the system prompt and user prompt from the task
            system_prompt = getattr(task, "system_prompt", None) or f"You are a {fb.agent_type} agent."
            user_prompt = getattr(task, "prompt", None)
            ai_response = getattr(task, "response", None)

            if not user_prompt or not ai_response:
                return None

            # For edited signals, prefer the user's edited version
            if fb.signal == "edited" and fb.user_edit:
                # user_edit contains {field: {original, edited}} diffs
                # Apply edits to the original response
                response = self._apply_edits(ai_response, fb.user_edit)
            else:
                response = ai_response

            return {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": response},
                ],
                "metadata": {
                    "agent_type": fb.agent_type,
                    "signal": fb.signal,
                    "quality_score": fb.quality_score,
                    "feedback_id": fb.id,
                },
            }
        except Exception as e:
            logger.warning(f"Failed to convert feedback {fb.id}: {e}")
            return None

    def _apply_edits(self, original: str, edits: dict) -> str:
        """Apply user edits to the original response."""
        try:
            # If the original is JSON, parse and apply field-level edits
            parsed = json.loads(original)
            if isinstance(parsed, dict):
                for field, diff in edits.items():
                    if isinstance(diff, dict) and "edited" in diff:
                        parsed[field] = diff["edited"]
                return json.dumps(parsed, ensure_ascii=False)
        except (json.JSONDecodeError, TypeError):
            pass

        # Fallback: return original unchanged
        return original

    async def export_jsonl(
        self,
        output_path: Optional[str] = None,
        since: Optional[datetime] = None,
        agent_type: Optional[str] = None,
    ) -> dict:
        """
        Export training data as JSONL file.

        Returns: {path: str, total_examples: int, by_agent: dict}
        """
        examples = await self.collect(since=since, agent_type=agent_type)

        if not examples:
            return {"path": None, "total_examples": 0, "by_agent": {}}

        EXPORT_DIR.mkdir(parents=True, exist_ok=True)
        path = Path(output_path) if output_path else EXPORT_DIR / f"sutra_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"

        by_agent: dict[str, int] = {}
        with open(path, "w") as f:
            for ex in examples:
                # Write only the messages (not metadata) to JSONL
                f.write(json.dumps({"messages": ex["messages"]}, ensure_ascii=False) + "\n")
                agent = ex["metadata"]["agent_type"]
                by_agent[agent] = by_agent.get(agent, 0) + 1

        result = {
            "path": str(path),
            "total_examples": len(examples),
            "by_agent": by_agent,
        }
        logger.info(f"TrainingDataCollector: exported {result}")
        return result
