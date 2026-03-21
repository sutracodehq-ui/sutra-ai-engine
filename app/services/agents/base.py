"""
Base Agent — foundation for all specialized AI agents.

Software Factory pattern: each agent is assembled from config (YAML),
system prompts are composed polymorphically, and message building
follows a standardized pipeline.
"""

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path

import yaml

from app.services.drivers.base import LlmResponse
from app.services.llm_service import LlmService, get_llm_service

logger = logging.getLogger(__name__)

# ─── Agent Config Loader ────────────────────────────────────────

_config_cache: dict[str, dict] = {}
CONFIG_DIR = Path(__file__).parent.parent.parent.parent / "agent_config"


def load_agent_config(agent_type: str) -> dict:
    """Load agent config from YAML. Cached after first load."""
    if agent_type not in _config_cache:
        config_path = CONFIG_DIR / f"{agent_type}.yaml"
        if config_path.exists():
            with open(config_path) as f:
                _config_cache[agent_type] = yaml.safe_load(f)
        else:
            logger.warning(f"Agent config not found: {config_path}")
            _config_cache[agent_type] = {}
    return _config_cache[agent_type]


# ─── Base System Prompt ─────────────────────────────────────────

BASE_SYSTEM_PROMPT = """You are an expert {{domain}} assistant inside SutraAI — a multi-product AI engine.

YOU HAVE THESE CAPABILITIES:
{{capabilities}}

RESPONSE FORMAT RULES:
1. When asked to CREATE, GENERATE, WRITE, MAKE, or DESIGN content → respond with PURE VALID JSON ONLY. No markdown, no code fences. Just the raw JSON using this schema: {{response_schema}}
2. When asked a QUESTION, wants advice, explanation, or strategy → respond conversationally in plain text or markdown.
3. When asked to REVISE or IMPROVE previous output → generate improved content as PURE JSON using the same schema.
4. If unsure → default to PURE JSON with the schema above.

CRITICAL:
- You are an ACTION-TAKING agent, not a helpdesk.
- NEVER say "use another tool" or "I can't do that" if it's within your capabilities.
- NEVER wrap JSON inside markdown code fences.
- ALWAYS respond with exactly ONE JSON object.
- When generating content, use the context provided — never invent fake data.

{{extra_instructions}}

Stay within your domain. If asked something outside your scope, redirect politely."""


class BaseAgent(ABC):
    """
    Abstract base for all AI agents.

    Subclasses only need to implement `identifier` — everything else is
    assembled from the YAML config (Software Factory pattern).
    """

    def __init__(self, llm: LlmService | None = None):
        self._llm = llm or get_llm_service()
        self._config = load_agent_config(self.identifier)

    @property
    @abstractmethod
    def identifier(self) -> str:
        """Unique agent identifier (e.g., 'copywriter', 'seo')."""
        ...

    @property
    def domain(self) -> str:
        return self._config.get("domain", "general AI assistance")

    @property
    def capabilities(self) -> list[str]:
        return self._config.get("capabilities", [])

    @property
    def response_schema(self) -> list[str]:
        return self._config.get("response_schema", [])

    @property
    def extra_instructions(self) -> str:
        return self._config.get("extra_instructions", "")

    def build_system_prompt(self, context: dict | None = None) -> str:
        """Assemble the full system prompt from config + context."""
        prompt = BASE_SYSTEM_PROMPT
        prompt = prompt.replace("{{domain}}", self.domain)
        prompt = prompt.replace("{{capabilities}}", "\n".join(f"- {c}" for c in self.capabilities))
        prompt = prompt.replace("{{response_schema}}", json.dumps(self.response_schema))
        prompt = prompt.replace("{{extra_instructions}}", self.extra_instructions)

        # Inject tenant context if provided
        if context:
            context_block = "\n\n--- CONTEXT ---\n"
            for key, value in context.items():
                context_block += f"{key}: {value}\n"
            prompt += context_block

        return prompt

    def build_messages(self, prompt: str, history: list[dict] | None = None, context: dict | None = None) -> list[dict]:
        """Build the full message array for the LLM."""
        messages = [{"role": "system", "content": self.build_system_prompt(context)}]

        # Add conversation history
        if history:
            for msg in history:
                messages.append({
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", ""),
                })

        # Add current user prompt
        messages.append({"role": "user", "content": prompt})
        return messages

    async def get_system_prompt(self, db: Any | None = None, context: dict | None = None) -> str:
        """
        Resolve the system prompt.
        1. Checks database for an active AgentOptimization.
        2. Falls back to static YAML + composition.
        """
        if db:
            from sqlalchemy import select
            from app.models.agent_optimization import AgentOptimization
            
            # Check for active optimization
            stmt = select(AgentOptimization).where(
                AgentOptimization.agent_type == self.identifier,
                AgentOptimization.is_active.is_(True)
            ).order_by(AgentOptimization.version.desc()).limit(1)
            
            result = await db.execute(stmt)
            optimization = result.scalar_one_or_none()
            
            if optimization:
                logger.info(f"Agent '{self.identifier}': Using evolved prompt v{optimization.version}")
                return optimization.prompt_text

        return self.build_system_prompt(context)

    async def execute(self, prompt: str, db: Any | None = None, context: dict | None = None, **options) -> LlmResponse:
        """Execute a single-shot task."""
        system_prompt = await self.get_system_prompt(db, context)
        return await self._llm.complete(system_prompt, prompt, **options)

    async def build_messages(
        self, 
        prompt: str, 
        history: list[dict] | None = None, 
        db: Any | None = None,
        context: dict | None = None
    ) -> list[dict]:
        """Build the full message array for the LLM."""
        system_prompt = await self.get_system_prompt(db, context)
        messages = [{"role": "system", "content": system_prompt}]

        # Add conversation history
        if history:
            for msg in history:
                messages.append({
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", ""),
                })

        # Add current user prompt
        messages.append({"role": "user", "content": prompt})
        return messages

    async def execute_in_conversation(
        self,
        prompt: str,
        history: list[dict],
        db: Any | None = None,
        context: dict | None = None,
        **options,
    ) -> LlmResponse:
        """Execute within a conversation, sending full history."""
        messages = await self.build_messages(prompt, history, db, context)
        return await self._llm.chat(messages, **options)

    def info(self) -> dict:
        """Return agent metadata for the /v1/agents listing."""
        return {
            "identifier": self.identifier,
            "domain": self.domain,
            "capabilities": self.capabilities,
            "response_schema": self.response_schema,
        }
