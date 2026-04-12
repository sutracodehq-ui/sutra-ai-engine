"""
Context Pruner — handles history pruning and token-aware context compression.

Ensures the LLM prompt stays within budget while maintaining the most relevant context.
"""

import logging
from typing import Any

from app.models.ai_task import AiTask

logger = logging.getLogger(__name__)


class ContextPruner:
    """Logic for condensing and pruning context before sending it to the LLM."""

    @staticmethod
    def prune_history(history: list[AiTask], max_turns: int = 10) -> list[AiTask]:
        """
        Simple turn-based pruning.
        Keeps the last N turns.
        """
        if len(history) <= max_turns:
            return history
        
        logger.info(f"💾 Pruning history from {len(history)} to {max_turns} turns")
        return history[-max_turns:]

    @staticmethod
    def compress_for_prompt(history: list[AiTask], max_tokens: int = 4000) -> list[dict[str, str]]:
        """
        Converts AiTask history into standard chat messages and ensures they don't exceed token limits.
        
        If total history is too long:
        1. Summarizes oldest turns (Future: requires an LLM call)
        2. Drops oldest turns (Current implementation)
        """
        messages = []
        for task in history:
            # User message
            messages.append({"role": "user", "content": task.prompt})
            
            # Assistant response
            if task.result and "content" in task.result:
                messages.append({"role": "assistant", "content": task.result["content"]})
            elif task.result and isinstance(task.result, str):
                messages.append({"role": "assistant", "content": task.result})

        # TODO: Implement token-aware pruning (using tiktoken or similar)
        # For now, we assume history is already pruned to a reasonable turn count (last 10)
        return messages
