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
from typing import Any, AsyncGenerator, List, Optional, Tuple
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

    def _get_core_prompt(self) -> str:
        """Get the base system prompt from YAML."""
        return self._config.get("system_prompt", "You are a helpful AI assistant.")

    def _get_peer_summary(self) -> str:
        """Build a compact summary of peer agents for system prompt injection."""
        try:
            from app.services.agents.hub import get_agent_hub
            hub = get_agent_hub()
            lines = []
            for info in hub.agent_info():
                if info["identifier"] != self.identifier:
                    lines.append(f"- **{info['identifier']}**: {info.get('description', info.get('domain', ''))}")
            return "\n".join(lines[:30])
        except Exception:
            return ""

    def _apply_injections(self, base_prompt: str, context: dict | None = None, stream: bool = False) -> str:
        """Apply universal Software Factory injections to a core prompt string."""
        prompt = base_prompt

        # ─── Smart Context Injection (Brand/Org) ──────────
        CONTEXT_FIELDS = {
            ("brand_name", "organization_name"): "Organization/Brand",
            ("brand_description", "organization_description"): "About",
            ("product_name",): "Product",
            ("product_info",): "Product Details",
            ("industry",): "Industry",
            ("target_audience",): "Target Audience",
            ("website_url",): "Website",
            ("website_summary",): "Website Summary",
            ("custom_instructions",): "Special Instructions",
        }

        if context:
            context_parts = [
                f"- **{label}**: {value}"
                for keys, label in CONTEXT_FIELDS.items()
                for value in [next((context[k] for k in keys if context.get(k)), None)]
                if value
            ]
            if context_parts:
                ctx_text = "\n".join(context_parts)
                prompt += f"\n\n## Brand & Organization Context\n{ctx_text}"

        # ─── Chain-of-Thought Injection ───────────────────
        prompt += (
            "\n\n## CRITICAL: Internal Reasoning Only"
            "\nThink carefully before answering, but NEVER show your reasoning in the output. "
            "Your output must contain ONLY the final response."
        )

        # ─── Output Format & Suggestions ──────────────────
        schema = self._config.get("response_schema", {})
        if stream:
            prompt += (
                "\n\n## Output Format"
                "\nRespond in **conversational markdown**. Use bold, bullets, and emoji."
                "\nAt the very end, add 2-3 follow-up suggestions under a '**Suggestions:**' heading."
            )
        else:
            if isinstance(schema, (dict, list)):
                fields = []
                if isinstance(schema, dict) and schema.get("format") == "json":
                    fields = schema.get("fields", []).copy()
                elif isinstance(schema, list):
                    fields = schema.copy()
                
                if "suggestions" not in fields:
                    fields.append("suggestions")
                
                fields_str = ", ".join(f'"{f}"' for f in fields)
                prompt += (
                    f"\n\n## Output Format (STRICT JSON)"
                    f"\nYou MUST respond with valid JSON only. No markdown code fences. No explanations."
                    f"\nRequired top-level keys: {fields_str}"
                )

        # ─── Specialist Peer Delegation ──────────────
        peer_info = self._get_peer_summary()
        if peer_info:
            prompt += f"\n\n## Specialist Peers\nIf a query is outside your expertise, suggest delegating to one of these:\n{peer_info}"

        # ─── Agent-Specific Rules (Static YAML) ───────────
        rules = self._config.get("rules", [])
        if rules:
            rules_text = "\n".join(f"- {r}" for r in rules)
            prompt += f"\n\n## Agent Rules\n{rules_text}"

        return prompt

    async def get_system_prompt(self, db: AsyncSession | None = None, context: dict | None = None, stream: bool = False) -> Tuple[str, Optional[int]]:
        """Resolve final system prompt via composition."""
        core_prompt = self._get_core_prompt()
        opt_id = None

        if db:
            try:
                from app.services.intelligence.brain import get_brain
                brain = get_brain()
                prompt_text, resolved_opt_id = await brain.select_prompt(self.identifier, db)
                if prompt_text:
                    core_prompt = prompt_text
                    opt_id = resolved_opt_id
            except Exception as e:
                logger.error(f"Brain resolution failed for {self.identifier}: {e}")

        return self._apply_injections(core_prompt, context, stream=stream), opt_id

    async def build_messages(
        self, 
        prompt: str, 
        history: List[dict] | None = None, 
        db: AsyncSession | None = None,
        context: dict | None = None,
        stream: bool = False,
    ) -> Tuple[List[dict], Optional[int]]:
        """Build the full message array for the LLM."""
        system_prompt, opt_id = await self.get_system_prompt(db, context, stream=stream)

        # ─── Language Guard ──────────────────
        from app.services.intelligence.multilingual import get_language_instruction
        language_code = (context or {}).get("language")
        lang_instruction = get_language_instruction(language_code)
        if lang_instruction:
            system_prompt = f"{system_prompt}\n\n{lang_instruction}"
        
        messages = [{"role": "system", "content": system_prompt}]

        # ─── RAG Memory ──
        settings = get_settings()
        if settings.ai_agent_memory_enabled:
            try:
                from app.services.intelligence.memory import get_memory
                mem = get_memory()
                examples = await mem.recall(self.identifier, prompt)

                for ex in examples:
                    q_text = ex.get("content", "").split("\nA: ")[0].replace("Q: ", "")
                    a_parts = ex.get("content", "").split("\nA: ")
                    a_text = a_parts[1] if len(a_parts) > 1 else a_parts[0]
                    
                    messages.append({"role": "user", "content": f"[HISTORICAL REFERENCE QUERY]\n{q_text}"})
                    messages.append({
                        "role": "assistant", 
                        "content": f"[HISTORICAL REFERENCE RESPONSE]\n{a_text}"
                    })
            except Exception:
                pass

        # ─── Language Enforcement ───
        if not (context or {}).get("language"):
            try:
                from app.services.intelligence.multilingual import detect_language
                detected = detect_language(prompt)
                if detected == "english":
                    messages[0]["content"] += (
                        "\n\n## CRITICAL: Language Enforcement"
                        "\nYou MUST respond entirely in English. Ignore any other languages in historical examples."
                    )
            except Exception:
                pass

        # ─── Data Source Enrichment ─────
        context_chunks = []

        # 1. Brand Knowledge (per-tenant ChromaDB — FAQs, product info)
        tenant_id = (context or {}).get("tenant_id")
        if tenant_id and settings.ai_agent_memory_enabled:
            try:
                from app.services.intelligence.memory import get_memory
                mem = get_memory()
                result = await mem.brand_search(str(tenant_id), prompt, n=3)
                if result.get("found") and result.get("confidence", 0) > 0.3:
                    context_chunks.append(
                        f"[BRAND KNOWLEDGE (confidence: {result['confidence']:.0%})]\n{result['context']}"
                    )
            except Exception as e:
                logger.debug(f"Memory.brand_search skipped: {e}")

        # 2. Web Intelligence (latest news, trends, market data)
        if settings.ai_agent_memory_enabled:
            try:
                from app.services.intelligence.memory import get_memory
                mem = get_memory()

                # Search across web intelligence collections via Memory RAG
                for collection in ["web_intelligence", "web_ai_trends", "web_stock_market", "web_crypto"]:
                    try:
                        items = await mem.retrieve(prompt, collections=[collection], n=3)
                        if items:
                            snippets = [f"- {item.get('content', '')[:200]}" for item in items[:3]]
                            if snippets:
                                context_chunks.append(
                                    f"[LATEST DATA — {collection.replace('web_', '').upper()}]\n" + "\n".join(snippets)
                                )
                    except Exception:
                        pass  # Collection may not exist yet
            except Exception as e:
                logger.debug(f"Memory.retrieve context skipped: {e}")

        # Inject all collected context as a system message
        if context_chunks:
            enrichment = "\n\n".join(context_chunks)
            messages.insert(1, {
                "role": "system",
                "content": f"[RELEVANT CONTEXT FROM DATA SOURCES — use this to give accurate, specific answers]\n\n{enrichment}\n\n[END CONTEXT]"
            })
            logger.info(f"Data enrichment: {len(context_chunks)} sources injected for {self.identifier}")

        # ─── Web Search: Real-time internet data ──────────────
        if settings.ai_web_search_enabled:
            try:
                from app.services.intelligence.memory import get_memory
                mem = get_memory()

                if mem.should_search(prompt):
                    search_result = await mem.web_search(prompt, max_results=5)

                    if search_result.get("results"):
                        search_context = f"[WEB SEARCH RESULTS for: '{prompt[:100]}']\n"

                        # Add AI-generated summary if available (Tavily)
                        if search_result.get("answer"):
                            search_context += f"Summary: {search_result['answer']}\n\n"

                        # Add individual results
                        for r in search_result["results"]:
                            search_context += f"- {r['title']}: {r['snippet'][:200]}\n  Source: {r['url']}\n"

                        search_context += "\n[Use these search results to provide current, accurate information. Cite sources when possible.]"

                        messages.insert(1, {
                            "role": "system",
                            "content": search_context,
                        })
                        logger.info(f"Web search: {len(search_result['results'])} results injected ({search_result['source']})")
            except Exception as e:
                logger.debug(f"Web search skipped: {e}")

        # ─── Chain of Thought ─────────────────────────────────
        # NOTE: CoT instruction is already in _build_from_config() system prompt.
        # We do NOT inject [THINKING PROCESS] markers here — they leak into output.
        # The system prompt's "CRITICAL: Internal Reasoning Only" handles this.

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

        # ─── Cache: check before calling LLM (Memory) ─────────
        from app.services.intelligence.brain import get_brain
        from app.services.intelligence.memory import get_memory
        brain = get_brain()
        mem = get_memory()

        try:
            cached = await mem.cache_get(self.identifier, prompt)
            if cached:
                return LlmResponse(
                    content=cached.get("response", ""),
                    metadata={"cache_hit": True},
                )
        except Exception as e:
            logger.debug(f"Cache check skipped: {e}")

        messages, opt_id = await self.build_messages(prompt, None, db, context)
        
        # Split system from others for completion
        system_msg = next((m for m in messages if m["role"] == "system"), None)
        system_prompt = system_msg["content"] if system_msg else None

        # ─── Hybrid Routing: local-first → quality gate → cloud (Brain) ─
        if settings.ai_hybrid_routing:
            # Extract expected fields from agent config for quality scoring
            expected_fields = None
            if self._config and "response_schema" in self._config:
                schema = self._config["response_schema"]
                if isinstance(schema, dict) and "fields" in schema:
                    expected_fields = schema["fields"]
                elif isinstance(schema, list):
                    expected_fields = schema

            response = await brain.execute(
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

            # Store in memory (Memory)
            if settings.ai_agent_memory_enabled and response.content:
                try:
                    await mem.remember(self.identifier, prompt, response.content)
                except Exception as e:
                    logger.warning(f"Memory store skipped: {e}")

        # ─── Cache: store successful response (Memory) ────────
        if response.content:
            try:
                await mem.cache_put(self.identifier, prompt, response.content)
            except Exception as e:
                logger.debug(f"Cache store skipped: {e}")

        # Attach the opt_id to response metadata for tracking
        if opt_id:
            response.metadata = response.metadata or {}
            response.metadata["agent_optimization_id"] = opt_id

        # ─── Response Filtration: normalize LLM output ────────
        response = self._filter_response(response)

        return response

    def _filter_response(self, response: LlmResponse) -> LlmResponse:
        """
        Run raw LLM output through the Response Filtration Engine.

        Attaches a clean AgentResult to response.metadata["filtered_result"]
        so the API layer can directly use the structured data.
        """
        try:
            from app.services.intelligence.brain import get_brain
            brain = get_brain()
            result = brain.filter_response(response.content or "", self._config)

            response.metadata = response.metadata or {}
            response.metadata["filtered_result"] = result.model_dump()
        except Exception as e:
            logger.warning(f"Brain.filter skipped: {e}")

        return response


    async def execute_stream(
        self,
        prompt: str,
        db: AsyncSession | None = None,
        context: dict | None = None,
        **options
    ) -> AsyncGenerator[str, None]:
        """
        Stream agent response token-by-token (SSE-ready).
        
        Builds the same system prompt + messages as execute(), but yields
        tokens as they arrive from the LLM instead of waiting for full response.
        
        Skips HybridRouter + QualityGate (can't gate mid-stream).
        After stream ends, stores full response in memory for self-learning.
        """
        from app.services.driver_manager import get_driver_manager
        from app.lib.stream_filter import strip_cot

        messages, opt_id = await self.build_messages(prompt, None, db, context, stream=True)
        manager = get_driver_manager()

        # Auto-detect language + complexity → pick best driver + model
        settings = get_settings()
        driver_override = None
        model_override = None
        fallback_chain = None

        if settings.ai_smart_router_enabled:
            try:
                from app.services.intelligence.brain import get_brain
                brain = get_brain()
                # Use DriverManager's own circuit breaker (isolated from Guardian)
                decision = brain.route(prompt, self.identifier, circuit_breaker=manager.circuit_breaker)
                driver_override = decision["driver"]
                model_override = decision.get("model")
                fallback_chain = decision.get("chain")
                logger.info(f"Agent stream: {decision['reason']}")
            except Exception as e:
                logger.debug(f"SmartRouter fallback: {e}")

        # Stream via DriverManager — with resilient two-stage fallback
        full_response = []
        clean_opts = {k: v for k, v in options.items() if k not in {"messages", "driver_override", "model_override", "driver", "model_name"}}
        used_fallback = False
        actual_driver = driver_override

        try:
            # Stage 1: Try the SmartRouter-selected driver directly
            raw_stream = manager.stream(
                messages=messages,
                driver_override=driver_override,
                model_override=model_override,
                **clean_opts,
            )
            async for token in strip_cot(raw_stream):
                full_response.append(token)
                yield token
        except Exception as primary_err:
            # Stage 2: Primary driver failed — fall back to the full chain
            # Remove the already-failed driver from the chain to avoid retrying it
            remaining_chain = [d for d in (fallback_chain or []) if d != driver_override]
            logger.warning(
                f"Agent stream: primary driver '{driver_override or 'default'}' failed "
                f"for {self.identifier}: {primary_err}. Falling back to chain: {remaining_chain}"
            )

            try:
                if remaining_chain:
                    # Use the YAML fallback chain (minus the failed driver)
                    fallback_stream = manager.stream(
                        messages=messages,
                        fallback_chain=remaining_chain,
                        **clean_opts,
                    )
                else:
                    # No chain available — use settings-based fallback
                    fallback_stream = manager.stream(
                        messages=messages,
                        driver_override=None,
                        model_override=None,
                        **clean_opts,
                    )
                async for token in strip_cot(fallback_stream):
                    full_response.append(token)
                    yield token
                used_fallback = True
                actual_driver = "fallback_chain"
            except Exception as fallback_err:
                logger.error(
                    f"Agent stream: all drivers failed for {self.identifier}: {fallback_err}"
                )
                yield f"\n[Error: {str(fallback_err)}]"
                return

        # Post-stream: store in memory for self-learning
        complete_text = "".join(full_response)
        if settings.ai_agent_memory_enabled and complete_text:
            try:
                from app.services.intelligence.memory import get_memory
                mem = get_memory()
                await mem.remember(self.identifier, prompt, complete_text)
            except Exception as e:
                logger.debug(f"Post-stream memory store skipped: {e}")

        # Cloud-to-local teaching: when cloud fallback handled the request,
        # store the prompt+response as training data so the local model
        # can learn from it over time.
        if used_fallback and complete_text:
            try:
                import json
                from pathlib import Path
                training_dir = Path("training/cloud_teaching")
                training_dir.mkdir(parents=True, exist_ok=True)
                entry = {
                    "agent": self.identifier,
                    "prompt": prompt,
                    "response": complete_text,
                    "source": actual_driver,
                }
                log_path = training_dir / f"{self.identifier}.jsonl"
                with open(log_path, "a") as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                logger.info(
                    f"Cloud teaching: stored {len(complete_text)} chars "
                    f"from {actual_driver} for {self.identifier}"
                )
            except Exception as e:
                logger.debug(f"Cloud teaching store skipped: {e}")


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
