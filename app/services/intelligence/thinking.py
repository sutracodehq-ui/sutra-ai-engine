"""
Thinking Middleware — chain-of-thought injection.

Performance impact: forces the LLM to reason step-by-step before answering.
Significantly improves quality for complex tasks (SEO analysis, strategy),
with minimal overhead for simple tasks (skipped via SmartRouter complexity).
"""

import logging

logger = logging.getLogger(__name__)


# Thinking prompt injected into the system message
THINKING_INJECTION = """
Before responding with the final JSON output, think through these steps internally:

1. UNDERSTAND: What exactly is being requested? What are the key requirements?
2. CONTEXT: What relevant information is available? What constraints exist?
3. APPROACH: What's the best strategy to produce high-quality output?
4. QUALITY CHECK: Is the output complete, accurate, and actionable?

After thinking through these steps, produce your final response.
Do NOT include your thinking process in the output — only the final result.
"""


class ThinkingMiddleware:
    """
    Injects chain-of-thought prompting for complex tasks.

    Activation:
    - Always active for "complex" tasks (per SmartRouter)
    - Skip for "simple" tasks (SMS, headlines, etc.)
    - Configurable per-agent via agent config
    """

    def __init__(self, *, enabled: bool = True):
        self._enabled = enabled

    def should_activate(self, complexity: str, agent_type: str) -> bool:
        """Determine if thinking should be injected."""
        if not self._enabled:
            return False

        # Always skip for simple tasks
        if complexity == "simple":
            return False

        # Always activate for complex, sometimes for moderate
        if complexity == "complex":
            return True

        # For moderate: activate for analytical agents only
        analytical_agents = {"seo", "email_campaign", "copywriter"}
        return agent_type in analytical_agents

    def inject(self, messages: list[dict], complexity: str, agent_type: str) -> list[dict]:
        """Inject thinking prompt into the message chain if appropriate."""
        if not self.should_activate(complexity, agent_type):
            return messages

        # Clone messages to avoid mutation
        messages = [m.copy() for m in messages]

        # Append thinking to the system prompt
        for msg in messages:
            if msg["role"] == "system":
                msg["content"] += THINKING_INJECTION
                logger.info(f"ThinkingMiddleware: injected for {agent_type} (complexity={complexity})")
                break

        return messages
