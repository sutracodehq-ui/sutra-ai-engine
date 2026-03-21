"""
Base Agent — the foundation for all specialized AI agents.

Software Factory:
- Registry: Agents are identified by a unique 'identifier'.
- Message Building: Agents handle the conversion of inputs to LLM-ready messages.
- Evolved Prompts: Agents resolve their system prompt from the DB (A/B testing) or YAML.
- Self-Learning: Agents recall similar past responses and store new ones via AgentMemoryService.
"""

import logging
import random
import yaml
from typing import Any, List, Optional, Tuple
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.llm_service import get_llm_service, LlmResponse
from app.config import get_settings

logger = logging.getLogger(__name__)


class BaseAgent:
    """Base class for all agents."""

    def __init__(self, llm=None):
        self._llm = llm or get_llm_service()
        # Software Factory: resolve config from identifier
        config_name = f"{self.identifier}.yaml"
        self.config_path = Path("agent_config") / config_name
        self._config = self._load_config()

    def _load_config(self) -> dict:
        """Load agent configuration from YAML."""
        if not self.config_path.exists():
            logger.warning(f"Config path {self.config_path} does not exist.")
            return {}
        with open(self.config_path, "r") as f:
            return yaml.safe_load(f)

    def _build_from_config(self, context: dict | None = None) -> str:
        """
        Build a system prompt from the static YAML config.
        
        Auto-injections:
        1. Chain-of-Thought: Adds reasoning instructions for better output quality
        2. JSON Mode: Enforces strict JSON output when response_schema.format == "json"
        3. Rules: Appends any agent-specific rules from YAML
        """
        prompt = self._config.get("system_prompt", "You are a helpful AI assistant.")

        # ─── Chain-of-Thought Injection ───────────────────
        cot_instruction = (
            "\n\n## Reasoning Process"
            "\nBefore generating your final response, think through the problem step by step:"
            "\n1. Analyze the user's request and identify key requirements"
            "\n2. Consider the best approach to fulfill each requirement"
            "\n3. Generate a high-quality, comprehensive response"
            "\nDo NOT include your reasoning steps in the final output."
        )
        prompt += cot_instruction

        # ─── JSON Mode Enforcement ────────────────────────
        schema = self._config.get("response_schema", {})
        if isinstance(schema, dict) and schema.get("format") == "json":
            fields = schema.get("fields", [])
            fields_str = ", ".join(f'"{f}"' for f in fields) if fields else "relevant fields"
            prompt += (
                f"\n\n## Output Format"
                f"\nYou MUST respond with valid JSON only. No markdown, no code fences, no explanations."
                f"\nRequired top-level keys: {fields_str}"
                f"\nEnsure all values are properly typed (strings, numbers, arrays as appropriate)."
            )
        elif isinstance(schema, list) and schema:
            fields_str = ", ".join(f'"{f}"' for f in schema)
            prompt += (
                f"\n\n## Output Format"
                f"\nYou MUST respond with valid JSON only. No markdown, no code fences, no explanations."
                f"\nRequired top-level keys: {fields_str}"
            )

        # ─── Rules Injection ──────────────────────────────
        rules = self._config.get("rules", [])
        if rules:
            rules_text = "\n".join(f"- {r}" for r in rules)
            prompt += f"\n\n## Rules\n{rules_text}"

        # ─── Capabilities Context ─────────────────────────
        capabilities = self._config.get("capabilities", [])
        if capabilities:
            caps_text = "\n".join(f"- {c}" for c in capabilities)
            prompt += f"\n\n## Your Capabilities\n{caps_text}"

        return prompt

    async def get_system_prompt(self, db: AsyncSession | None = None, context: dict | None = None) -> Tuple[str, Optional[int]]:
        """
        Resolve the system prompt via the Self-Optimizing Prompt Engine.
        
        Strategy:
        1. PromptEngine selects champion or candidate (with explore rate)
        2. Fallback to YAML config if no DB prompts exist
        
        Returns (prompt_text, optimization_id).
        """
        if not db:
            return self._build_from_config(context), None

        try:
            from app.services.intelligence.prompt_engine import PromptEngine
            engine = PromptEngine(db)

            prompt_text, opt_id = await engine.select_prompt(self.identifier)
            if prompt_text:
                return prompt_text, opt_id

        except Exception as e:
            logger.error(f"PromptEngine fallback for {self.identifier}: {e}")

        # Final fallback: Static YAML with CoT/JSON injections
        return self._build_from_config(context), None


    async def build_messages(
        self, 
        prompt: str, 
        history: List[dict] | None = None, 
        db: AsyncSession | None = None,
        context: dict | None = None
    ) -> Tuple[List[dict], Optional[int]]:
        """Build the full message array for the LLM."""
        system_prompt, opt_id = await self.get_system_prompt(db, context)

        # ─── Inject Multilingual Support ──────────────────
        from app.services.intelligence.multilingual import get_language_instruction
        language_code = (context or {}).get("language")  # e.g. "hi", "mai", "bho"
        lang_instruction = get_language_instruction(language_code)
        if lang_instruction:
            system_prompt = f"{system_prompt}\n\n{lang_instruction}"

        messages = [{"role": "system", "content": system_prompt}]

        # ─── Self-Learning: Inject RAG Memory ─────────────
        settings = get_settings()
        if settings.ai_agent_memory_enabled:
            try:
                from app.services.intelligence.agent_memory import get_agent_memory
                memory = get_agent_memory()
                examples = await memory.recall(self.identifier, prompt)
                for ex in examples:
                    messages.append({"role": "user", "content": ex["prompt"]})
                    messages.append({"role": "assistant", "content": ex["response"]})
            except Exception as e:
                logger.warning(f"AgentMemory recall skipped: {e}")

        # Add conversation history
        if history:
            for msg in history:
                messages.append({
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", ""),
                })

        # Add current user prompt
        messages.append({"role": "user", "content": prompt})
        return messages, opt_id

    async def execute(
        self, 
        prompt: str, 
        db: AsyncSession | None = None, 
        context: dict | None = None, 
        **options
    ) -> LlmResponse:
        """Execute a single-shot task via HybridRouter or direct LLM."""

        settings = get_settings()

        # ─── Response Cache: check before calling LLM ─────────
        try:
            from app.services.intelligence.response_cache import get_response_cache
            cache = get_response_cache()
            cached = await cache.get(self.identifier, prompt)
            if cached:
                return LlmResponse(
                    content=cached["content"],
                    metadata={**(cached.get("metadata", {})), "cache_hit": True},
                )
        except Exception as e:
            logger.debug(f"Cache check skipped: {e}")

        messages, opt_id = await self.build_messages(prompt, None, db, context)
        
        # Split system from others for completion
        system_msg = next((m for m in messages if m["role"] == "system"), None)
        system_prompt = system_msg["content"] if system_msg else None

        # ─── Hybrid Routing: local-first → quality gate → cloud ─────
        if settings.ai_hybrid_routing:
            from app.services.intelligence.hybrid_router import get_hybrid_router
            router = get_hybrid_router()

            # Extract expected fields from agent config for quality scoring
            expected_fields = None
            if self.config and "response_schema" in self.config:
                schema = self.config["response_schema"]
                if isinstance(schema, dict) and "fields" in schema:
                    expected_fields = schema["fields"]
                elif isinstance(schema, list):
                    expected_fields = schema

            response = await router.execute(
                prompt=prompt,
                system_prompt=system_prompt,
                agent_type=self.identifier,
                expected_fields=expected_fields,
                **options
            )
        else:
            # ─── Direct mode (hybrid disabled) ─────────────────
            service = get_llm_service()
            response = await service.complete(
                prompt=prompt,
                system_prompt=system_prompt,
                **options
            )

            # Store in memory when not using hybrid (hybrid handles its own)
            if settings.ai_agent_memory_enabled and response.content:
                try:
                    from app.services.intelligence.agent_memory import get_agent_memory
                    memory = get_agent_memory()
                    await memory.remember(self.identifier, prompt, response.content)
                except Exception as e:
                    logger.warning(f"AgentMemory store skipped: {e}")

        # ─── Response Cache: store successful response ────────
        if response.content:
            try:
                from app.services.intelligence.response_cache import get_response_cache
                cache = get_response_cache()
                await cache.put(self.identifier, prompt, response.content)
            except Exception as e:
                logger.debug(f"Cache store skipped: {e}")

        # Attach the opt_id to response metadata for tracking
        if opt_id:
            response.metadata = response.metadata or {}
            response.metadata["agent_optimization_id"] = opt_id

        return response



    async def execute_in_conversation(
        self,
        prompt: str,
        history: List[dict],
        db: AsyncSession | None = None,
        context: dict | None = None,
        **options
    ) -> LlmResponse:
        """Execute within a conversation, sending full history."""
        messages, opt_id = await self.build_messages(prompt, history, db, context)
        
        # Extract system prompt
        system_msg = next((m for m in messages if m["role"] == "system"), None)
        system_prompt = system_msg["content"] if system_msg else None
        other_messages = [m for m in messages if m["role"] != "system"]
        
        service = get_llm_service()
        response = await service.chat(
            messages=other_messages,
            system_prompt=system_prompt,
            **options
        )

        if opt_id:
            response.metadata = response.metadata or {}
            response.metadata["agent_optimization_id"] = opt_id

        return response

    def info(self) -> dict:
        """Return agent metadata for the /v1/agents listing."""
        # Software Factory: extract fields list from response_schema dict
        resp_schema = self._config.get("response_schema", {})
        if isinstance(resp_schema, dict):
            resp_fields = resp_schema.get("fields", [])
        else:
            resp_fields = []

        return {
            "identifier": self.identifier,
            "name": self._config.get("name", self.identifier.replace("_", " ").title()),
            "domain": self._config.get("domain", "general"),
            "description": self._config.get("description", ""),
            "capabilities": self._config.get("capabilities", []),
            "response_schema": resp_fields,
        }
