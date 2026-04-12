"""
Domain-Specific Evolution Engine — Vertical-aware model benchmarking.

Software Factory Principle: Continuous Improvement + Domain Expertise.

Extends the base EvolutionEngine with domain-aware capabilities:
1. DOMAIN BENCHMARK: Test models with domain-specific prompts
   (finance agents tested on finance prompts, not generic ones)
2. DOMAIN SELF-TEACH: Generate training data from domain-relevant prompts
3. DOMAIN EVOLVE: Full evolution cycle scoped to a vertical

The engine reads domain benchmarks from intelligence_config.yaml
and runs focused evolution cycles per vertical.
"""

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml

from app.config import get_settings
from app.services.intelligence.evolution_engine import EvolutionEngine, get_evolution_engine

logger = logging.getLogger(__name__)

CONFIG_PATH = Path("intelligence_config.yaml")


def _load_domain_config() -> dict:
    """Load domain benchmarks config from YAML."""
    if not CONFIG_PATH.exists():
        return {}
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f) or {}
    return config.get("domain_benchmarks", {})


def _load_teaching_alliances() -> dict:
    """Load teaching alliances to map domains → agents."""
    if not CONFIG_PATH.exists():
        return {}
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f) or {}
    return config.get("cross_teaching", {}).get("alliances", {})


class DomainEvolution:
    """
    Domain-specific evolution engine.

    Instead of benchmarking all agents with generic prompts like
    "Explain photosynthesis", this engine uses domain-relevant prompts:
    - Finance agents tested on ROI calculations and risk analysis
    - Health agents tested on triage scenarios and lab reports
    - EdTech agents tested on quiz generation and lecture planning

    This produces more meaningful quality signals per vertical.
    """

    def __init__(self):
        self._base_engine = get_evolution_engine()
        self._history_path = Path("training/domain_evolution_history.jsonl")
        self._history_path.parent.mkdir(parents=True, exist_ok=True)

    # ─── Domain Benchmark ───────────────────────────────────

    async def benchmark_domain(self, domain: str, model_name: str | None = None) -> dict:
        """
        Benchmark a model using domain-specific test prompts.

        Unlike the generic benchmark, this uses prompts that match
        the domain's actual use cases.
        """
        domain_config = _load_domain_config()
        domain_prompts = domain_config.get(domain, [])

        if not domain_prompts:
            logger.warning(f"DomainEvolution: no benchmarks configured for domain '{domain}'")
            return {"domain": domain, "status": "no_config"}

        settings = get_settings()
        model = model_name or settings.ollama_model

        results = {
            "domain": domain,
            "model": model,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_prompts": len(domain_prompts),
            "passed": 0,
            "failed": 0,
            "avg_latency_ms": 0,
            "avg_quality": 0,
            "details": [],
        }

        total_latency = 0
        total_quality = 0

        for bp in domain_prompts:
            start = time.monotonic()
            try:
                import httpx
                async with httpx.AsyncClient(timeout=60) as client:
                    resp = await client.post(
                        f"{settings.ollama_url}/api/generate",
                        json={
                            "model": model,
                            "prompt": bp["prompt"],
                            "stream": False,
                            "options": {"num_predict": bp.get("max_tokens", 300)},
                        },
                    )
                latency = (time.monotonic() - start) * 1000

                if resp.status_code != 200:
                    results["failed"] += 1
                    results["details"].append({"prompt": bp["prompt"][:60], "status": "http_error"})
                    continue

                data = resp.json()
                content = data.get("response", "")

                # Score using the base engine's scoring logic
                quality = self._base_engine._score_response(content, bp)
                total_latency += latency
                total_quality += quality

                detail = {
                    "prompt": bp["prompt"][:60],
                    "latency_ms": round(latency),
                    "quality": quality,
                    "content_length": len(content),
                    "status": "pass" if quality >= 6 else "fail",
                }
                results["details"].append(detail)

                if quality >= 6:
                    results["passed"] += 1
                else:
                    results["failed"] += 1

            except Exception as e:
                results["failed"] += 1
                results["details"].append({"prompt": bp["prompt"][:60], "status": f"error: {e}"})

        n = results["passed"] + results["failed"]
        results["avg_latency_ms"] = round(total_latency / max(n, 1))
        results["avg_quality"] = round(total_quality / max(n, 1), 2)
        results["pass_rate"] = round(results["passed"] / max(n, 1) * 100, 1)

        logger.info(
            f"DomainEvolution: benchmark [{domain}] {model} → "
            f"quality={results['avg_quality']}, pass_rate={results['pass_rate']}%"
        )
        return results

    # ─── Domain Self-Teach ──────────────────────────────────

    async def self_teach_domain(self, domain: str, model_name: str | None = None) -> dict:
        """
        Generate domain-specific training data.

        Uses domain benchmark prompts to generate high-quality
        training examples, producing better LoRA data than generic prompts.
        """
        domain_config = _load_domain_config()
        domain_prompts = domain_config.get(domain, [])

        if not domain_prompts:
            return {"domain": domain, "status": "no_config", "generated": 0}

        settings = get_settings()
        model = model_name or settings.ollama_model
        output_dir = Path("training/data")
        output_dir.mkdir(parents=True, exist_ok=True)

        results = {"domain": domain, "model": model, "generated": 0, "failed": 0, "examples": []}
        output_file = output_dir / f"domain_teach_{domain}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"

        # Get alliance info for domain-specific system prompts
        alliances = _load_teaching_alliances()
        domain_topics = alliances.get(domain, {}).get("shared_topics", [])
        topics_context = f" Focus areas: {', '.join(domain_topics)}." if domain_topics else ""

        for bp in domain_prompts:
            try:
                import httpx
                async with httpx.AsyncClient(timeout=60) as client:
                    resp = await client.post(
                        f"{settings.ollama_url}/api/generate",
                        json={
                            "model": model,
                            "prompt": bp["prompt"],
                            "system": (
                                f"You are a domain expert in {domain}.{topics_context} "
                                "Provide clear, comprehensive, well-structured responses."
                            ),
                            "stream": False,
                            "options": {"temperature": 0.7, "num_predict": bp.get("max_tokens", 400)},
                        },
                    )

                if resp.status_code == 200:
                    content = resp.json().get("response", "")
                    if content and len(content) > 50:
                        example = {
                            "messages": [
                                {"role": "system", "content": f"You are a {domain} domain expert."},
                                {"role": "user", "content": bp["prompt"]},
                                {"role": "assistant", "content": content},
                            ],
                            "domain": domain,
                        }
                        results["examples"].append(example)
                        results["generated"] += 1
                    else:
                        results["failed"] += 1
                else:
                    results["failed"] += 1

            except Exception as e:
                results["failed"] += 1
                logger.warning(f"DomainEvolution: self-teach failed for {domain}: {e}")

        # Write training data
        if results["examples"]:
            with open(output_file, "w") as f:
                for ex in results["examples"]:
                    f.write(json.dumps(ex, ensure_ascii=False) + "\n")
            logger.info(
                f"DomainEvolution: domain self-teach [{domain}] "
                f"generated {results['generated']} examples → {output_file}"
            )

        return results

    # ─── Full Domain Evolution Cycle ────────────────────────

    async def evolve_domain(self, domain: str) -> dict:
        """
        Run a complete evolution cycle for a specific domain.

        1. Benchmark current model on domain prompts
        2. Benchmark candidates on domain prompts
        3. Recommend upgrade if domain performance improves
        4. Self-teach from best model using domain data
        """
        settings = get_settings()
        current_model = settings.ollama_model
        evolution_config = self._base_engine._EvolutionEngine__dict__ if hasattr(self._base_engine, '__dict__') else {}

        cycle_result = {
            "domain": domain,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "current_model": current_model,
            "current_benchmark": None,
            "candidates_tested": 0,
            "best_candidate": None,
            "self_teach_examples": 0,
        }

        logger.info(f"DomainEvolution: starting [{domain}] evolution (model: {current_model})")

        # 1. Benchmark current model on domain
        current_bench = await self.benchmark_domain(domain, current_model)
        cycle_result["current_benchmark"] = {
            "quality": current_bench.get("avg_quality", 0),
            "pass_rate": current_bench.get("pass_rate", 0),
        }

        # 2. Self-teach from domain data
        teach_result = await self.self_teach_domain(domain, current_model)
        cycle_result["self_teach_examples"] = teach_result.get("generated", 0)

        # 3. Log history
        self._log_history(cycle_result)

        logger.info(
            f"DomainEvolution: [{domain}] cycle complete — "
            f"quality={current_bench.get('avg_quality', 0)}, "
            f"self_taught={cycle_result['self_teach_examples']}"
        )
        return cycle_result

    # ─── Run All Domains ────────────────────────────────────

    async def evolve_all_domains(self) -> dict:
        """Run domain evolution for all configured domains."""
        domain_config = _load_domain_config()
        if not domain_config:
            return {"status": "no_domains_configured"}

        results = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "domains": {},
        }

        for domain in domain_config:
            try:
                result = await self.evolve_domain(domain)
                results["domains"][domain] = {
                    "quality": result.get("current_benchmark", {}).get("quality", 0),
                    "self_taught": result.get("self_teach_examples", 0),
                }
            except Exception as e:
                logger.error(f"DomainEvolution: error evolving {domain}: {e}")
                results["domains"][domain] = {"error": str(e)}

        return results

    def _log_history(self, cycle_result: dict) -> None:
        """Append domain evolution result to history log."""
        try:
            with open(self._history_path, "a") as f:
                f.write(json.dumps(cycle_result, ensure_ascii=False, default=str) + "\n")
        except Exception as e:
            logger.warning(f"DomainEvolution: failed to log history: {e}")


# ─── Singleton ──────────────────────────────────────────────
_domain_engine: DomainEvolution | None = None


def get_domain_evolution() -> DomainEvolution:
    global _domain_engine
    if _domain_engine is None:
        _domain_engine = DomainEvolution()
    return _domain_engine
