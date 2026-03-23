"""
Hybrid Intelligence Router — Local-first, cloud-escalation, auto-train.

The brain of the SutraAI self-learning pipeline:
1. Try local model (Qwen 2.5:3b) first → zero cost, fast
2. Quality-gate the response → if good, return immediately
3. If bad → escalate to cloud (Groq → Gemini fallback)
4. Cloud's good response auto-trains the local model via AgentMemory
5. Per-agent quality tracking adapts routing over time

Result: The more you use cloud, the smarter local gets, the less you need cloud.
"""

import logging
import time
from typing import Optional

from app.config import get_settings
from app.services.drivers.base import LlmResponse
from app.services.intelligence.quality_gate import QualityGate
from app.services.intelligence.quality_tracker import get_quality_tracker

logger = logging.getLogger(__name__)


class HybridRouter:
    """
    Hybrid local/cloud router with automatic local model training.

    Routing strategies based on per-agent quality history:
    - fast_local: Agent consistently scores >8 → trust local, skip quality gate
    - standard: Try local → quality gate → escalate to cloud if needed
    - direct_cloud: Agent consistently scores <5 → go straight to cloud
    """

    def __init__(self):
        self._quality_gate = QualityGate(
            enabled=True,
            threshold=get_settings().ai_hybrid_quality_threshold,
        )
        self._tracker = get_quality_tracker()

    async def execute(
        self,
        prompt: str,
        system_prompt: str,
        agent_type: str,
        expected_fields: list[str] | None = None,
        **options
    ) -> LlmResponse:
        """
        Execute with hybrid routing: local-first → quality gate → cloud escalation.

        Returns the best response and auto-trains local from cloud when escalated.
        """
        settings = get_settings()

        if not settings.ai_hybrid_routing:
            # Hybrid disabled — fall through to default driver
            return await self._call_driver(prompt, system_prompt, **options)

        # Get routing hint from quality history
        route_hint = await self._tracker.get_route_hint(agent_type)
        logger.info(f"HybridRouter: agent={agent_type}, route_hint={route_hint}")

        if route_hint == "direct_cloud":
            # Local is consistently bad for this agent → skip to cloud
            response = await self._call_cloud(prompt, system_prompt, _agent_type=agent_type, **options)
            await self._auto_train(agent_type, prompt, response)
            return response

        # ─── Step 1: Try Local ─────────────────────────────
        start = time.monotonic()
        local_response = await self._call_local(prompt, system_prompt, **options)
        local_latency = round((time.monotonic() - start) * 1000)

        if route_hint == "fast_local":
            # Agent has proven track record → trust local
            logger.info(f"HybridRouter: FAST_LOCAL for {agent_type} ({local_latency}ms)")
            await self._tracker.record(agent_type, 8.0)  # Maintain fast path
            return local_response

        # ─── Step 2: Quality Gate ──────────────────────────
        quality = self._quality_gate.score(local_response, expected_fields)
        local_score = quality["total"]
        logger.info(f"HybridRouter: local score={local_score} (threshold={quality['threshold']})")

        if quality["passed"]:
            # Local is good enough → return it, record success
            await self._tracker.record(agent_type, local_score)
            logger.info(f"HybridRouter: LOCAL_PASS for {agent_type} ({local_latency}ms, score={local_score})")
            return local_response

        # ─── Step 3: Escalate to Cloud ─────────────────────
        logger.info(f"HybridRouter: ESCALATING {agent_type} to cloud (local_score={local_score})")
        cloud_response = await self._call_cloud(prompt, system_prompt, _agent_type=agent_type, **options)

        # Quality-check cloud response too
        cloud_quality = self._quality_gate.score(cloud_response, expected_fields)
        cloud_score = cloud_quality["total"]

        # Record the local failure score for routing adaptation
        await self._tracker.record(agent_type, local_score)

        if cloud_quality["passed"]:
            # Cloud succeeded — auto-train local from this response
            await self._auto_train(agent_type, prompt, cloud_response)
            logger.info(
                f"HybridRouter: CLOUD_WIN for {agent_type} "
                f"(local={local_score}, cloud={cloud_score}) → auto-training local"
            )
            return cloud_response

        # Both failed — return the better one
        if cloud_score > local_score:
            return cloud_response
        return local_response

    async def _call_local(self, prompt: str, system_prompt: str, **options) -> LlmResponse:
        """Call the local Ollama model."""
        from app.services.llm_service import get_llm_service
        service = get_llm_service()
        try:
            return await service.complete(
                prompt=prompt,
                system_prompt=system_prompt,
                driver="ollama",
                **options
            )
        except Exception as e:
            logger.warning(f"HybridRouter: local call failed: {e}")
            # Return a dummy response so escalation can proceed
            return LlmResponse(
                content="",
                total_tokens=0,
                driver="ollama",
                model="qwen2.5:3b",
                metadata={"error": str(e)},
            )

    async def _call_cloud(self, prompt: str, system_prompt: str, **options) -> LlmResponse:
        """
        Call cloud with tiered escalation + SmartRouter model selection:
        Tier 1: Groq (free, fast — Llama 3.3 70B)
        Tier 2: Gemini (cheap, smart — Flash or Pro based on complexity)
        Tier 3: Anthropic (premium, smartest — Haiku or Sonnet based on complexity)
        """
        from app.services.llm_service import get_llm_service
        service = get_llm_service()

        # SmartRouter: pick optimal model per driver based on complexity
        agent_type = options.pop("_agent_type", "unknown")
        model_overrides = {}
        settings = get_settings()
        if settings.ai_smart_router_enabled:
            try:
                from app.services.intelligence.smart_router import SmartRouter
                smart = SmartRouter(enabled=True)
                for driver_name in ["groq", "gemini", "anthropic"]:
                    model = smart.select_model(prompt, agent_type, driver_name)
                    if model:
                        model_overrides[driver_name] = model
            except Exception as e:
                logger.debug(f"HybridRouter: SmartRouter skipped: {e}")

        # Ordered by cost: free → cheap → premium
        cloud_tiers = [
            ("groq", "Tier1:Free"),
            ("gemini", "Tier2:Smart"),
            ("anthropic", "Tier3:Premium"),
        ]

        for driver, tier_label in cloud_tiers:
            try:
                model_override = model_overrides.get(driver)
                response = await service.complete(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    driver=driver,
                    model=model_override,
                    **options
                )
                response.metadata = response.metadata or {}
                response.metadata["cloud_tier"] = tier_label
                if model_override:
                    response.metadata["smart_model"] = model_override
                logger.info(
                    f"HybridRouter: cloud success via {tier_label} ({driver}"
                    f"{f', model={model_override}' if model_override else ''})"
                )
                return response
            except Exception as e:
                logger.warning(f"HybridRouter: {tier_label} ({driver}) failed: {e}")
                continue

        # All cloud drivers failed — return empty
        return LlmResponse(
            content="",
            total_tokens=0,
            driver="none",
            model="none",
            metadata={"error": "all cloud drivers failed"},
        )

    async def _call_driver(self, prompt: str, system_prompt: str, **options) -> LlmResponse:
        """Default call — uses whatever driver is configured."""
        from app.services.llm_service import get_llm_service
        service = get_llm_service()
        return await service.complete(
            prompt=prompt,
            system_prompt=system_prompt,
            **options
        )

    async def _auto_train(self, agent_type: str, prompt: str, response: LlmResponse) -> None:
        """
        Auto-train: store cloud's good response for local model to learn from.

        Two storage paths:
        1. AgentMemory (ChromaDB) → immediate recall on similar future prompts
        2. TrainingCollector (JSONL) → periodic LoRA fine-tune
        """
        if not response.content:
            return

        try:
            from app.services.intelligence.agent_memory import get_agent_memory
            memory = get_agent_memory()
            await memory.remember(
                agent_type=agent_type,
                prompt=prompt,
                response=response.content,
                quality_score=1.0,  # Cloud-validated, high quality
            )
            logger.info(f"HybridRouter: auto-trained {agent_type} from cloud response")
        except Exception as e:
            logger.warning(f"HybridRouter: auto-train failed: {e}")


# ─── Singleton ──────────────────────────────────────────────
_router: HybridRouter | None = None


def get_hybrid_router() -> HybridRouter:
    """Get the global HybridRouter instance."""
    global _router
    if _router is None:
        _router = HybridRouter()
    return _router
