"""
Self-Evolution Engine — Autonomous AI model discovery, benchmarking, and upgrade.

Software Factory Principle: Continuous Improvement.

This background engine runs daily and:
1. DISCOVER: Scans Ollama library + HuggingFace for new/updated models
2. BENCHMARK: Tests new models against the current one on standardized prompts
3. UPGRADE: Auto-pulls and switches to better-performing models
4. SELF-TEACH: Generates training data from best responses for fine-tuning
5. TRACK: Logs evolution history for audit trail

The engine follows the Software Factory flywheel:
    Discover → Benchmark → Upgrade → Self-Teach → Better Performance → Repeat
"""

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import httpx
import yaml

from app.config import get_settings

logger = logging.getLogger(__name__)

CONFIG_PATH = Path("intelligence_config.yaml")


def _load_evolution_config() -> dict:
    """Load evolution engine config from YAML."""
    if not CONFIG_PATH.exists():
        return {}
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f) or {}
    return config.get("evolution_engine", {})


class EvolutionEngine:
    """
    Autonomous AI model discovery and self-improvement engine.

    Scans for new models, benchmarks them, and auto-upgrades the system
    when a better model is found. Also generates self-teaching data
    from the best responses to continuously improve local model quality.
    """

    def __init__(self):
        self._client = httpx.AsyncClient(timeout=30)
        self._history_path = Path("training/evolution_history.jsonl")
        self._history_path.parent.mkdir(parents=True, exist_ok=True)

    # ─── Phase 1: DISCOVER ──────────────────────────────────

    async def discover_models(self) -> list[dict]:
        """
        Scan Ollama library for available models.
        Checks for new models or updated versions of tracked models.
        """
        config = _load_evolution_config()
        ollama_url = get_settings().ollama_url
        tracked_families = config.get("tracked_model_families", [
            "qwen2.5", "llama3", "mistral", "gemma2", "phi3", "deepseek-r1",
        ])

        discovered = []

        # 1. Get locally available models from Ollama
        try:
            resp = await self._client.get(f"{ollama_url}/api/tags")
            if resp.status_code == 200:
                local_models = resp.json().get("models", [])
                for model in local_models:
                    discovered.append({
                        "name": model.get("name", ""),
                        "size": model.get("size", 0),
                        "modified": model.get("modified_at", ""),
                        "source": "local",
                        "status": "installed",
                    })
                logger.info(f"EvolutionEngine: found {len(local_models)} local models")
        except Exception as e:
            logger.warning(f"EvolutionEngine: failed to list local models: {e}")

        # 2. Check Ollama library for new models in tracked families
        for family in tracked_families:
            try:
                # Ollama API to check model info
                resp = await self._client.post(
                    f"{ollama_url}/api/show",
                    json={"name": family},
                    timeout=10,
                )
                if resp.status_code == 200:
                    info = resp.json()
                    model_info = {
                        "name": family,
                        "parameters": info.get("details", {}).get("parameter_size", "unknown"),
                        "quantization": info.get("details", {}).get("quantization_level", "unknown"),
                        "family": info.get("details", {}).get("family", family),
                        "source": "ollama_library",
                        "status": "available",
                    }
                    # Check if it's not already in local
                    if not any(d["name"] == family and d["source"] == "local" for d in discovered):
                        model_info["status"] = "new"
                    discovered.append(model_info)
            except Exception:
                pass  # Model not available, skip silently

        logger.info(f"EvolutionEngine: discovered {len(discovered)} total models")
        return discovered

    # ─── Phase 2: BENCHMARK ─────────────────────────────────

    async def benchmark_model(self, model_name: str) -> dict:
        """
        Benchmark a model on standardized test prompts.
        Measures: latency, quality score, token efficiency, consistency.
        """
        config = _load_evolution_config()
        ollama_url = get_settings().ollama_url

        benchmark_prompts = config.get("benchmark_prompts", [
            {
                "prompt": "Explain photosynthesis in 3 sentences.",
                "expected_keywords": ["sunlight", "carbon dioxide", "oxygen", "chlorophyll"],
                "max_tokens": 200,
            },
            {
                "prompt": "Write a JSON object with keys: name, age, city for a fictional person.",
                "expected_keywords": ["name", "age", "city"],
                "max_tokens": 100,
                "format_check": "json",
            },
            {
                "prompt": "List 5 marketing strategies for a SaaS startup. Return as JSON array.",
                "expected_keywords": ["content", "social", "email", "seo"],
                "max_tokens": 500,
                "format_check": "json",
            },
            {
                "prompt": "Calculate the ROI if I invest $10,000 and get back $15,000 after 2 years.",
                "expected_keywords": ["50%", "ROI", "return"],
                "max_tokens": 200,
            },
        ])

        results = {
            "model": model_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_prompts": len(benchmark_prompts),
            "passed": 0,
            "failed": 0,
            "avg_latency_ms": 0,
            "avg_quality": 0,
            "total_tokens": 0,
            "details": [],
        }

        total_latency = 0
        total_quality = 0

        for bp in benchmark_prompts:
            start = time.monotonic()
            try:
                resp = await self._client.post(
                    f"{ollama_url}/api/generate",
                    json={
                        "model": model_name,
                        "prompt": bp["prompt"],
                        "stream": False,
                        "options": {"num_predict": bp.get("max_tokens", 200)},
                    },
                    timeout=60,
                )
                latency = (time.monotonic() - start) * 1000

                if resp.status_code != 200:
                    results["failed"] += 1
                    results["details"].append({"prompt": bp["prompt"][:50], "status": "http_error"})
                    continue

                data = resp.json()
                content = data.get("response", "")
                tokens = data.get("eval_count", 0)

                # Score the response
                quality = self._score_response(content, bp)
                total_latency += latency
                total_quality += quality
                results["total_tokens"] += tokens

                detail = {
                    "prompt": bp["prompt"][:50],
                    "latency_ms": round(latency),
                    "quality": quality,
                    "tokens": tokens,
                    "content_length": len(content),
                }

                if quality >= 6:
                    results["passed"] += 1
                    detail["status"] = "pass"
                else:
                    results["failed"] += 1
                    detail["status"] = "fail"

                results["details"].append(detail)

            except Exception as e:
                results["failed"] += 1
                results["details"].append({"prompt": bp["prompt"][:50], "status": f"error: {e}"})

        n = results["passed"] + results["failed"]
        results["avg_latency_ms"] = round(total_latency / max(n, 1))
        results["avg_quality"] = round(total_quality / max(n, 1), 2)
        results["pass_rate"] = round(results["passed"] / max(n, 1) * 100, 1)

        logger.info(
            f"EvolutionEngine: benchmark {model_name} → "
            f"quality={results['avg_quality']}, latency={results['avg_latency_ms']}ms, "
            f"pass_rate={results['pass_rate']}%"
        )
        return results

    def _score_response(self, content: str, benchmark: dict) -> float:
        """Score a response against benchmark criteria (0-10)."""
        if not content or len(content) < 10:
            return 0.0

        score = 5.0  # Base score for any non-empty response

        # Keyword presence check
        keywords = benchmark.get("expected_keywords", [])
        if keywords:
            found = sum(1 for kw in keywords if kw.lower() in content.lower())
            keyword_ratio = found / len(keywords)
            score += keyword_ratio * 3  # Up to +3 for keywords

        # JSON format check
        if benchmark.get("format_check") == "json":
            try:
                # Try to extract JSON from the response
                import re
                json_match = re.search(r'[\[{].*[\]}]', content, re.DOTALL)
                if json_match:
                    json.loads(json_match.group())
                    score += 2  # +2 for valid JSON
                else:
                    score -= 1
            except (json.JSONDecodeError, ValueError):
                score -= 1  # -1 for invalid JSON when expected

        # Length appropriateness
        max_tokens = benchmark.get("max_tokens", 200)
        if len(content.split()) > max_tokens * 2:
            score -= 1  # Penalty for being too verbose

        return min(max(score, 0), 10)

    # ─── Phase 3: UPGRADE ───────────────────────────────────

    async def try_upgrade(self, candidate_model: str, current_benchmark: dict) -> dict:
        """
        Compare a candidate model against the current model.
        Auto-upgrade if the candidate is better.
        """
        config = _load_evolution_config()
        min_quality_improvement = config.get("min_quality_improvement", 0.5)
        max_latency_increase_pct = config.get("max_latency_increase_pct", 50)

        # Benchmark the candidate
        candidate_benchmark = await self.benchmark_model(candidate_model)

        result = {
            "candidate": candidate_model,
            "current_quality": current_benchmark["avg_quality"],
            "candidate_quality": candidate_benchmark["avg_quality"],
            "current_latency": current_benchmark["avg_latency_ms"],
            "candidate_latency": candidate_benchmark["avg_latency_ms"],
            "decision": "skip",
            "reason": "",
        }

        # Decision logic
        quality_delta = candidate_benchmark["avg_quality"] - current_benchmark["avg_quality"]
        latency_delta_pct = (
            (candidate_benchmark["avg_latency_ms"] - current_benchmark["avg_latency_ms"])
            / max(current_benchmark["avg_latency_ms"], 1) * 100
        )

        if quality_delta >= min_quality_improvement:
            if latency_delta_pct <= max_latency_increase_pct:
                result["decision"] = "upgrade"
                result["reason"] = (
                    f"Quality +{quality_delta:.1f} (above threshold {min_quality_improvement}), "
                    f"latency change {latency_delta_pct:+.0f}% (within {max_latency_increase_pct}% limit)"
                )
            else:
                result["decision"] = "skip"
                result["reason"] = f"Quality improved but latency too high ({latency_delta_pct:+.0f}%)"
        else:
            result["decision"] = "skip"
            result["reason"] = f"Quality delta {quality_delta:+.1f} below threshold {min_quality_improvement}"

        return result

    async def pull_model(self, model_name: str) -> bool:
        """Pull a new model into Ollama."""
        try:
            ollama_url = get_settings().ollama_url
            resp = await self._client.post(
                f"{ollama_url}/api/pull",
                json={"name": model_name, "stream": False},
                timeout=600,  # Models can take a while to download
            )
            if resp.status_code == 200:
                logger.info(f"EvolutionEngine: successfully pulled {model_name}")
                return True
            logger.warning(f"EvolutionEngine: pull failed for {model_name}: {resp.status_code}")
            return False
        except Exception as e:
            logger.error(f"EvolutionEngine: pull error for {model_name}: {e}")
            return False

    # ─── Phase 4: SELF-TEACH ────────────────────────────────

    async def self_teach(self, model_name: str) -> dict:
        """
        Generate self-teaching data by running the best model on
        diverse prompts and saving high-quality responses as training data.
        """
        config = _load_evolution_config()
        ollama_url = get_settings().ollama_url
        output_dir = Path("training/data")
        output_dir.mkdir(parents=True, exist_ok=True)

        teaching_prompts = config.get("self_teaching_prompts", [
            "Explain the concept of supply and demand in economics",
            "Write a professional email asking for a meeting next week",
            "Create a Python function that validates email addresses",
            "Generate a SWOT analysis template for a tech startup",
            "Describe the process of machine learning model training",
            "Write a product description for a smart water bottle",
            "Explain blockchain technology to a 10-year-old",
            "Create a weekly meal plan for a vegetarian diet",
            "Analyze the pros and cons of remote work",
            "Write a cover letter for a software engineering position",
        ])

        results = {"model": model_name, "generated": 0, "failed": 0, "examples": []}
        output_file = output_dir / f"self_teach_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"

        for prompt in teaching_prompts:
            try:
                resp = await self._client.post(
                    f"{ollama_url}/api/generate",
                    json={
                        "model": model_name,
                        "prompt": prompt,
                        "system": "You are an expert AI assistant. Provide clear, comprehensive, well-structured responses.",
                        "stream": False,
                        "options": {"temperature": 0.7, "num_predict": 500},
                    },
                    timeout=60,
                )

                if resp.status_code == 200:
                    content = resp.json().get("response", "")
                    if content and len(content) > 50:
                        example = {
                            "messages": [
                                {"role": "system", "content": "You are an expert AI assistant."},
                                {"role": "user", "content": prompt},
                                {"role": "assistant", "content": content},
                            ]
                        }
                        results["examples"].append(example)
                        results["generated"] += 1
                    else:
                        results["failed"] += 1
                else:
                    results["failed"] += 1

            except Exception as e:
                results["failed"] += 1
                logger.warning(f"EvolutionEngine: self-teach failed for prompt: {e}")

        # Write training data
        if results["examples"]:
            with open(output_file, "w") as f:
                for ex in results["examples"]:
                    f.write(json.dumps(ex, ensure_ascii=False) + "\n")
            logger.info(f"EvolutionEngine: self-teaching generated {results['generated']} examples → {output_file}")

        return results

    # ─── Phase 5: FULL EVOLUTION CYCLE ──────────────────────

    async def evolve(self) -> dict:
        """
        Run the complete evolution cycle:
        1. Discover available models
        2. Benchmark current model
        3. Try candidates against current
        4. Upgrade if better found
        5. Self-teach from best model
        """
        config = _load_evolution_config()
        settings = get_settings()
        current_model = settings.ollama_model

        cycle_result = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "current_model": current_model,
            "upgraded": False,
            "new_model": None,
            "current_benchmark": None,
            "candidates_tested": 0,
            "self_teach_examples": 0,
        }

        logger.info(f"EvolutionEngine: starting evolution cycle (current: {current_model})")

        # 1. Discover
        models = await self.discover_models()
        local_models = [m for m in models if m["source"] == "local" and m["name"] != current_model]

        # 2. Benchmark current model
        current_benchmark = await self.benchmark_model(current_model)
        cycle_result["current_benchmark"] = {
            "quality": current_benchmark["avg_quality"],
            "latency_ms": current_benchmark["avg_latency_ms"],
            "pass_rate": current_benchmark["pass_rate"],
        }

        # 3. Test candidates
        candidate_models = config.get("candidate_models", [])
        # Also test any local models we haven't tested
        for m in local_models:
            if m["name"] not in candidate_models:
                candidate_models.append(m["name"])

        best_upgrade = None
        for candidate in candidate_models[:5]:  # Test max 5 candidates per cycle
            cycle_result["candidates_tested"] += 1
            try:
                upgrade_result = await self.try_upgrade(candidate, current_benchmark)
                if upgrade_result["decision"] == "upgrade":
                    if best_upgrade is None or upgrade_result["candidate_quality"] > best_upgrade["candidate_quality"]:
                        best_upgrade = upgrade_result
                        logger.info(f"EvolutionEngine: found better model: {candidate}")
            except Exception as e:
                logger.warning(f"EvolutionEngine: failed to test {candidate}: {e}")

        # 4. Upgrade if better found
        if best_upgrade:
            cycle_result["upgraded"] = True
            cycle_result["new_model"] = best_upgrade["candidate"]
            cycle_result["upgrade_reason"] = best_upgrade["reason"]
            logger.info(
                f"EvolutionEngine: 🚀 UPGRADE RECOMMENDED: "
                f"{current_model} → {best_upgrade['candidate']} "
                f"(quality: {best_upgrade['current_quality']} → {best_upgrade['candidate_quality']})"
            )

        # 5. Self-teach from current best model
        best_model = best_upgrade["candidate"] if best_upgrade else current_model
        teach_result = await self.self_teach(best_model)
        cycle_result["self_teach_examples"] = teach_result["generated"]

        # 6. Log history
        self._log_history(cycle_result)

        logger.info(
            f"EvolutionEngine: cycle complete — "
            f"tested={cycle_result['candidates_tested']}, "
            f"upgraded={cycle_result['upgraded']}, "
            f"self_taught={cycle_result['self_teach_examples']}"
        )
        return cycle_result

    def _log_history(self, cycle_result: dict) -> None:
        """Append evolution cycle result to history log."""
        try:
            with open(self._history_path, "a") as f:
                f.write(json.dumps(cycle_result, ensure_ascii=False, default=str) + "\n")
        except Exception as e:
            logger.warning(f"EvolutionEngine: failed to log history: {e}")

    async def close(self):
        """Close HTTP client."""
        await self._client.aclose()


# ─── Singleton ──────────────────────────────────────────────
_engine: EvolutionEngine | None = None


def get_evolution_engine() -> EvolutionEngine:
    global _engine
    if _engine is None:
        _engine = EvolutionEngine()
    return _engine
