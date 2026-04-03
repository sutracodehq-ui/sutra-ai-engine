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

import json
import logging
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from app.config import get_settings
from app.services.drivers.base import LlmResponse
from app.services.intelligence.quality_engine import QualityEngine, get_quality_engine
from app.services.intelligence.driver import get_driver_registry

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
        self._engine = QualityEngine(
            enabled=True,
            threshold=get_settings().ai_hybrid_quality_threshold,
        )

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

        if not get_settings().ai_hybrid_routing:
            # Hybrid disabled — fall through to default driver
            return await self._call_driver(prompt, system_prompt, **options)

        # ─── New: A/B Testing Gate ─────────────────────────
        try:
            from app.services.intelligence.config_loader import get_intelligence_config
            cfg = get_intelligence_config()
            ab = cfg.get("smart_router", {}).get("ab_testing", {})
            
            if ab.get("enabled", False):
                agent_ab = ab.get("agents", {}).get(agent_type, {})
                split = agent_ab.get("split", ab.get("default_split", 0.0))
                
                if random.random() < split:
                    tuned_model = agent_ab.get("tuned_model")
                    if tuned_model:
                        logger.info(f"HybridRouter: AB_TEST (tuned) for {agent_type} -> {tuned_model}")
                        options["model"] = tuned_model
                        # When testing tuned model, we skip the normal hybrid flow to see how it performs solo
                        return await self._call_local(prompt, system_prompt, **options)
        except Exception as e:
            logger.debug(f"HybridRouter: AB_TEST skipped: {e}")

        # Get routing hint from quality history
        route_hint = await self._engine.get_route_hint(agent_type)
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
            await self._engine.record(agent_type, 8.0)  # Maintain fast path
            return local_response

        # ─── Step 2: Quality Gate ──────────────────────────
        quality = self._engine.score(local_response, expected_fields)
        local_score = quality["total"]
        logger.info(f"HybridRouter: local score={local_score} (threshold={quality['threshold']})")

        if quality["passed"]:
            # Local is good enough → return it, record success
            await self._engine.record(agent_type, local_score)
            logger.info(f"HybridRouter: LOCAL_PASS for {agent_type} ({local_latency}ms, score={local_score})")
            return local_response

        # ─── Step 3: Escalate to Cloud ─────────────────────
        logger.info(f"HybridRouter: ESCALATING {agent_type} to cloud (local_score={local_score})")
        
        # Select cloud driver via SmartRouter or Tiered escalation
        cloud_response = await self._call_cloud(prompt, system_prompt, _agent_type=agent_type, **options)

        # Quality-check cloud response too
        cloud_quality = self._engine.score(cloud_response, expected_fields)
        cloud_score = cloud_quality["total"]

        # Record the local failure score for routing adaptation
        await self._engine.record(agent_type, local_score)

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
            # Even if cloud "failed" quality gate, if it's better than local, we should log it for distillation
            if cloud_score > local_score + 1.0:
                 await self._save_to_training_log(agent_type, prompt, cloud_response, "cloud_win")
            return cloud_response
        return local_response

    async def _call_local(self, prompt: str, system_prompt: str, **options) -> LlmResponse:
        """Call the local Ollama model via Registry."""
        registry = get_driver_registry()

        # ── Guard: skip immediately if Ollama circuit is OPEN ──
        if not registry.circuit_breaker.is_available("ollama"):
            logger.warning("HybridRouter: local skipped — Ollama circuit OPEN")
            return LlmResponse(
                content="", total_tokens=0, driver="ollama", model="qwen2.5:3b",
                metadata={"error": "circuit_open"},
            )

        try:
            return await registry.complete(
                system_prompt=system_prompt,
                user_prompt=prompt,
                driver_override="ollama",
                **options
            )
        except Exception as e:
            logger.warning(f"HybridRouter: local call failed: {e}")
            return LlmResponse(
                content="", total_tokens=0, driver="ollama", model="qwen2.5:3b",
                metadata={"error": str(e)},
            )

    async def _call_cloud(self, prompt: str, system_prompt: str, **options) -> LlmResponse:
        """Call cloud with tiered escalation via Registry."""
        registry = get_driver_registry()

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
                response = await registry.complete(
                    system_prompt=system_prompt,
                    user_prompt=prompt,
                    driver_override=driver,
                    model_override=model_override,
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
        """Default call via Registry."""
        registry = get_driver_registry()
        return await registry.complete(
            system_prompt=system_prompt,
            user_prompt=prompt,
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
            # 1. Store in Memory
            from app.services.intelligence.agent_memory import get_agent_memory
            memory = get_agent_memory()
            await memory.remember(
                agent_type=agent_type,
                prompt=prompt,
                response=response.content,
                quality_score=1.0,  # Cloud-validated, high quality
            )
            
            # 2. Log for LoRA Training
            await self._save_to_training_log(agent_type, prompt, response, "distillation")
            
            logger.info(f"HybridRouter: auto-trained {agent_type} from cloud response")
        except Exception as e:
            logger.warning(f"HybridRouter: auto-train failed: {e}")

    async def _save_to_training_log(self, agent_type: str, prompt: str, response: LlmResponse, source: str) -> None:
        """Save a high-quality (prompt, response) pair for LoRA training."""
        try:
            from app.services.intelligence.config_loader import get_intelligence_config
            cfg = get_intelligence_config()
            distill_cfg = cfg.get("distillation", {})
            
            if not distill_cfg.get("enabled", True):
                return

            save_path = Path(distill_cfg.get("save_path", "training/data/per_agent"))
            save_path.mkdir(parents=True, exist_ok=True)
            
            log_file = save_path / f"{agent_type}.jsonl"
            
            # Create training example in chat format
            example = {
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response.content}
                ],
                "metadata": {
                    "agent_type": agent_type,
                    "source": source,
                    "model": response.model,
                    "driver": response.driver,
                    "timestamp": datetime.now().isoformat()
                }
            }
            
            # Append in a single atomic write (approx)
            with open(log_file, "a") as f:
                f.write(json.dumps(example, ensure_ascii=False) + "\n")
                
        except Exception as e:
            logger.warning(f"HybridRouter: failed to save training log: {e}")


# ─── Singleton ──────────────────────────────────────────────
_router: HybridRouter | None = None


def get_hybrid_router() -> HybridRouter:
    """Get the global HybridRouter instance."""
    global _router
    if _router is None:
        _router = HybridRouter()
    return _router
