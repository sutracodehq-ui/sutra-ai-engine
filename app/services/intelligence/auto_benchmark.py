"""
Auto Benchmark Suite — Dynamic quality tracking from real user queries.

Software Factory Principle: Quality Control + Continuous Improvement.

Unlike static benchmarks with fixed prompts, this suite:
1. Captures real user queries as benchmark candidates
2. Tracks per-agent quality trends over time
3. Auto-generates improvement suggestions when quality drops
4. Compares agent performance across model versions
"""

import json
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import yaml

from app.config import get_settings

logger = logging.getLogger(__name__)


def _load_benchmark_config() -> dict:
    """Load benchmark config from YAML."""
    config_path = Path("intelligence_config.yaml")
    if not config_path.exists():
        return {}
    with open(config_path) as f:
        config = yaml.safe_load(f) or {}
    return config.get("auto_benchmark", {})


class AutoBenchmark:
    """
    Dynamic benchmark suite that generates and runs benchmarks
    from real user queries rather than static test prompts.
    """

    def __init__(self):
        self._benchmark_dir = Path("training/benchmarks")
        self._benchmark_dir.mkdir(parents=True, exist_ok=True)
        self._history_path = self._benchmark_dir / "benchmark_history.jsonl"

    # ─── Capture Real Queries ───────────────────────────────

    async def capture_query(
        self,
        agent_type: str,
        prompt: str,
        response: str,
        quality_score: float,
    ) -> None:
        """
        Capture a real user query as a potential benchmark.
        Only high-quality query-response pairs become benchmarks.
        """
        config = _load_benchmark_config()
        min_quality = config.get("min_quality_for_benchmark", 7.0)

        if quality_score < min_quality:
            return

        benchmark_file = self._benchmark_dir / f"captured_{agent_type}.jsonl"
        entry = {
            "agent_type": agent_type,
            "prompt": prompt,
            "expected_response_preview": response[:300],
            "quality_score": quality_score,
            "captured_at": datetime.now(timezone.utc).isoformat(),
        }

        with open(benchmark_file, "a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # ─── Run Benchmarks ─────────────────────────────────────

    async def run_agent_benchmark(self, agent_type: str) -> dict:
        """
        Run captured benchmarks against an agent and track quality.
        """
        benchmark_file = self._benchmark_dir / f"captured_{agent_type}.jsonl"

        if not benchmark_file.exists():
            return {"status": "no_benchmarks", "agent_type": agent_type}

        # Load captured benchmarks
        benchmarks = []
        with open(benchmark_file) as f:
            for line in f:
                if line.strip():
                    benchmarks.append(json.loads(line))

        if not benchmarks:
            return {"status": "no_benchmarks", "agent_type": agent_type}

        # Run each benchmark
        from app.services.agents.hub import AiAgentHub
        hub = AiAgentHub()
        agent = hub.get(agent_type)

        if not agent:
            return {"status": "agent_not_found", "agent_type": agent_type}

        config = _load_benchmark_config()
        max_benchmarks = config.get("max_benchmarks_per_run", 10)

        results = {
            "agent_type": agent_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_benchmarks": min(len(benchmarks), max_benchmarks),
            "scores": [],
            "avg_quality": 0,
            "pass_rate": 0,
        }

        total_score = 0
        passed = 0

        for bm in benchmarks[:max_benchmarks]:
            try:
                response = await agent.execute(bm["prompt"])

                # Score the response
                score = self._score_benchmark(response.content, bm)
                results["scores"].append({
                    "prompt_preview": bm["prompt"][:80],
                    "score": score,
                    "passed": score >= 6,
                })
                total_score += score
                if score >= 6:
                    passed += 1

            except Exception as e:
                results["scores"].append({
                    "prompt_preview": bm["prompt"][:80],
                    "score": 0,
                    "passed": False,
                    "error": str(e)[:100],
                })

        n = results["total_benchmarks"]
        results["avg_quality"] = round(total_score / max(n, 1), 2)
        results["pass_rate"] = round(passed / max(n, 1) * 100, 1)

        # Log history
        with open(self._history_path, "a") as f:
            f.write(json.dumps(results, ensure_ascii=False) + "\n")

        logger.info(
            f"AutoBenchmark: {agent_type} → quality={results['avg_quality']}, "
            f"pass_rate={results['pass_rate']}%"
        )
        return results

    def _score_benchmark(self, response: str, benchmark: dict) -> float:
        """Score a benchmark response."""
        if not response or len(response) < 20:
            return 0.0

        score = 5.0

        # Length check (not too short, not too verbose)
        if len(response) > 100:
            score += 1
        if len(response) > 500:
            score += 1

        # JSON validity for structured agents
        try:
            import re
            if re.search(r'[\[{]', response):
                json_match = re.search(r'[\[{].*[\]}]', response, re.DOTALL)
                if json_match:
                    json.loads(json_match.group())
                    score += 2
        except (json.JSONDecodeError, AttributeError):
            pass

        return min(score, 10)

    # ─── Trend Analysis ─────────────────────────────────────

    async def get_quality_trends(self, agent_type: str, days: int = 30) -> dict:
        """Analyze quality trends over time for an agent."""
        if not self._history_path.exists():
            return {"agent_type": agent_type, "trend": "no_data"}

        entries = []
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)

        with open(self._history_path) as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    if entry.get("agent_type") == agent_type:
                        entry_time = datetime.fromisoformat(
                            entry["timestamp"].replace("Z", "+00:00")
                        )
                        if entry_time >= cutoff:
                            entries.append(entry)

        if len(entries) < 2:
            return {"agent_type": agent_type, "trend": "insufficient_data"}

        # Calculate trend
        scores = [e["avg_quality"] for e in entries]
        first_half = scores[:len(scores)//2]
        second_half = scores[len(scores)//2:]

        avg_first = sum(first_half) / len(first_half)
        avg_second = sum(second_half) / len(second_half)
        trend_direction = "improving" if avg_second > avg_first else "declining"
        trend_delta = round(avg_second - avg_first, 2)

        result = {
            "agent_type": agent_type,
            "period_days": days,
            "data_points": len(entries),
            "current_avg": round(avg_second, 2),
            "previous_avg": round(avg_first, 2),
            "trend": trend_direction,
            "trend_delta": trend_delta,
        }

        # Generate alert if declining
        if trend_direction == "declining" and abs(trend_delta) > 0.5:
            result["alert"] = (
                f"⚠️ {agent_type} quality declining by {abs(trend_delta):.1f} points. "
                f"Consider reviewing prompt or model."
            )

        return result

    # ─── Run All Agents ─────────────────────────────────────

    async def run_all(self) -> list[dict]:
        """Run benchmarks for all agents that have captured data."""
        results = []
        for bm_file in self._benchmark_dir.glob("captured_*.jsonl"):
            agent_type = bm_file.stem.replace("captured_", "")
            result = await self.run_agent_benchmark(agent_type)
            results.append(result)
        return results


# ─── Singleton ──────────────────────────────────────────────
_benchmark: AutoBenchmark | None = None


def get_auto_benchmark() -> AutoBenchmark:
    global _benchmark
    if _benchmark is None:
        _benchmark = AutoBenchmark()
    return _benchmark
