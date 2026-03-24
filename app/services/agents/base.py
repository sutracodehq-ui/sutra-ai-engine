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

    def _build_from_config(self, context: dict | None = None, stream: bool = False) -> str:
        """
        Build a system prompt from the static YAML config.
        
        Auto-injections:
        1. Brand/Org Context: Injects brand, product, and organization info when available
        2. Chain-of-Thought: Adds reasoning instructions for better output quality
        3. Output Format: JSON for REST (/run), markdown for SSE (/stream)
        4. Rules: Appends any agent-specific rules from YAML
        """
        prompt = self._config.get("system_prompt", "You are a helpful AI assistant.")

        # ─── Smart Context Injection (Polymorphic — config-driven) ────
        # Map: (context_key_or_fallback, display_label)
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
                prompt += (
                    f"\n\n## Brand & Organization Context"
                    f"\nYou are representing this brand/organization. Use this information to give specific, "
                    f"contextual responses — never give generic advice that ignores this context."
                    f"\n{ctx_text}"
                )

        # ─── Chain-of-Thought Injection ───────────────────
        cot_instruction = (
            "\n\n## CRITICAL: Internal Reasoning Only"
            "\nThink carefully before answering, but NEVER show your reasoning "
            "in the output. Do NOT include phrases like 'Let me think', "
            "'THINKING PROCESS', 'Step 1:', or any internal reasoning. "
            "Your output must contain ONLY the final, polished response "
            "— nothing else."
        )
        prompt += cot_instruction

        # ─── Output Format (mode-aware) ───────────────────
        schema = self._config.get("response_schema", {})

        if stream:
            # STREAM mode → conversational markdown for real-time display
            prompt += (
                "\n\n## Output Format"
                "\nRespond in **conversational markdown** — NOT JSON."
                "\nUse bold headings, bullet points, numbered steps, and emoji for clarity."
                "\nDo NOT wrap your response in ```json``` code blocks."
                "\nAt the very end of your response, add 2-3 follow-up suggestions "
                "as a bullet list under a '**Suggestions:**' heading."
            )
        else:
            # RUN mode → structured JSON for programmatic consumption
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

        # ─── Peer Awareness (Collaboration) ───────────────
        peer_info = self._get_peer_summary()
        if peer_info:
            delegate_note = (
                'include a "delegate_to" key in your JSON response'
                if not stream else
                "suggest delegating to the appropriate agent"
            )
            prompt += (
                "\n\n## Collaboration — Your Specialist Peers"
                "\nYou have access to these specialist agents. If a user's query is outside your expertise, "
                f"{delegate_note} with the agent identifier."
                "\nOnly delegate if the query is clearly better handled by another agent."
                f"\n{peer_info}"
            )

        # ─── Proactive Suggestions ────────────────────────
        if stream:
            prompt += (
                "\n\n## Proactive Suggestions"
                "\nAt the end of your response, under '**Suggestions:**', include 2-3 specific, "
                "actionable follow-up questions or next steps the user can take."
            )
        else:
            prompt += (
                "\n\n## Proactive Suggestions"
                '\nAlways include a "suggestions" key in your JSON response. This must be an array of 2-3 specific, actionable '
                "follow-up questions or next steps the user can take."
            )

        # ─── Re-enforce Output Format (JSON only) ─────────
        if not stream and isinstance(schema, dict) and schema.get("format") == "json":
            fields = schema.get("fields", []).copy()
            if "suggestions" not in fields:
                fields.append("suggestions")
            fields_str = ", ".join(f'"{f}"' for f in fields)
            prompt += (
                f"\n\n## Refined Output Format"
                f"\nYou MUST respond with valid JSON only. No markdown, no code fences, no explanations."
                f"\nRequired top-level keys: {fields_str}"
            )

        return prompt

    def _get_peer_summary(self) -> str:
        """Build a compact summary of peer agents for system prompt injection."""
        try:
            from app.services.agents.hub import get_agent_hub
            hub = get_agent_hub()
            lines = []
            for info in hub.agent_info():
                if info["identifier"] != self.identifier:
                    lines.append(f"- **{info['identifier']}**: {info.get('description', info.get('domain', ''))}")
            # Limit to 30 peers to avoid prompt bloat
            return "\n".join(lines[:30])
        except Exception:
            return ""

    async def get_system_prompt(self, db: AsyncSession | None = None, context: dict | None = None, stream: bool = False) -> Tuple[str, Optional[int]]:
        """
        Resolve the system prompt via the Self-Optimizing Prompt Engine.
        
        Strategy:
        1. PromptEngine selects champion or candidate (with explore rate)
        2. Fallback to YAML config if no DB prompts exist
        
        Returns (prompt_text, optimization_id).
        """
        if not db:
            return self._build_from_config(context, stream=stream), None

        try:
            from app.services.intelligence.prompt_engine import PromptEngine
            engine = PromptEngine(db)

            prompt_text, opt_id = await engine.select_prompt(self.identifier)
            if prompt_text:
                return prompt_text, opt_id

        except Exception as e:
            logger.error(f"PromptEngine fallback for {self.identifier}: {e}")

        # Final fallback: Static YAML with CoT/JSON injections
        return self._build_from_config(context, stream=stream), None


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

                # Cross-agent learning: if no own memories, check peers
                if not examples:
                    cross_examples = await memory.recall_cross_agent(self.identifier, prompt)
                    for ex in cross_examples:
                        source = ex.get("source_agent", "peer")
                        messages.append({"role": "user", "content": ex["prompt"]})
                        messages.append({"role": "assistant", "content": f"[Insight from {source}]: {ex['response']}"})
                else:
                    for ex in examples:
                        messages.append({"role": "user", "content": ex["prompt"]})
                        messages.append({"role": "assistant", "content": ex["response"]})
            except Exception as e:
                logger.warning(f"AgentMemory recall skipped: {e}")

        # ─── Data Source Enrichment: Brand Knowledge + Web Intel ─────
        context_chunks = []

        # 1. Brand Knowledge (per-tenant ChromaDB — FAQs, product info)
        tenant_id = (context or {}).get("tenant_id")
        if tenant_id:
            try:
                from app.services.intelligence.brand_knowledge import get_brand_knowledge
                bk = get_brand_knowledge()
                result = await bk.search(str(tenant_id), prompt, n_results=3)
                if result.get("found") and result.get("confidence", 0) > 0.3:
                    context_chunks.append(
                        f"[BRAND KNOWLEDGE (confidence: {result['confidence']:.0%})]\n{result['context']}"
                    )
            except Exception as e:
                logger.debug(f"BrandKnowledge search skipped: {e}")

        # 2. Web Intelligence (latest news, trends, market data)
        try:
            from app.services.intelligence.web_scanner import get_web_scanner
            scanner = get_web_scanner()

            # Search across web intelligence collections
            for collection in ["web_intelligence", "web_ai_trends", "web_stock_market", "web_crypto"]:
                try:
                    items = await scanner.get_context(prompt, collection_name=collection, n_results=3)
                    if items:
                        snippets = []
                        for item in items[:3]:
                            title = item.get("title", item.get("symbol", ""))
                            desc = item.get("description", item.get("name", ""))
                            if title:
                                snippets.append(f"- {title}: {desc}")
                        if snippets:
                            context_chunks.append(
                                f"[LATEST DATA — {collection.replace('web_', '').upper()}]\n" + "\n".join(snippets)
                            )
                except Exception:
                    pass  # Collection may not exist yet
        except Exception as e:
            logger.debug(f"WebScanner context skipped: {e}")

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
                from app.services.intelligence.web_search import get_web_search
                ws = get_web_search()

                if ws.should_search(prompt):
                    search_result = await ws.search(prompt, max_results=5)

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
            if self._config and "response_schema" in self._config:
                schema = self._config["response_schema"]
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
            from app.services.intelligence.response_filter import get_response_filter
            engine = get_response_filter()
            result = engine.filter(response.content or "", self._config)

            response.metadata = response.metadata or {}
            response.metadata["filtered_result"] = result.model_dump()
        except Exception as e:
            logger.warning(f"ResponseFilter skipped: {e}")

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

        # Auto-detect language + complexity → pick best driver + model
        settings = get_settings()
        driver_override = None
        model_override = None

        if settings.ai_smart_router_enabled:
            try:
                from app.services.intelligence.smart_router import SmartRouter
                cb = get_driver_manager().circuit_breaker
                decision = SmartRouter(enabled=True).route(prompt, self.identifier, circuit_breaker=cb)
                driver_override = decision["driver"]
                model_override = decision.get("model")
                logger.info(f"Agent stream: {decision['reason']}")
            except Exception as e:
                logger.debug(f"SmartRouter fallback: {e}")

        # Stream via DriverManager — with resilient fallback
        manager = get_driver_manager()
        full_response = []
        clean_opts = {k: v for k, v in options.items() if k not in {"messages", "driver_override", "model_override", "driver", "model_name"}}
        used_fallback = False
        actual_driver = driver_override

        try:
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
            # Primary driver failed — fall back to cloud via the full chain
            logger.warning(
                f"Agent stream: primary driver '{driver_override or 'default'}' failed "
                f"for {self.identifier}: {primary_err}. Falling back to chain."
            )

            try:
                # Retry WITHOUT driver_override → uses the full fallback chain
                fallback_stream = manager.stream(
                    messages=messages,
                    driver_override=None,  # let the chain handle it
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
                from app.services.intelligence.agent_memory import get_agent_memory
                memory = get_agent_memory()
                await memory.remember(self.identifier, prompt, complete_text)
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
