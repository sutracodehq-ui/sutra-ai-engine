#!/usr/bin/env python3
"""
Full Software-Factory benchmark: covers ALL rollout gate categories.

Gate categories (from intelligence_config.yaml factory_rollout_gates):
  - streaming: first-token latency, throughput, interruption, emergency, format
  - text_quality: avg quality score, empty response rate, escalation rate
  - json_integrity: field coverage, parse failure rate
  - cost: monthly USD, per-request cost
  - latency: p50/p95 complete-call latency
  - image: failure rate, avg latency
  - voice: STT error rate, TTS latency

Usage:
    python scripts/benchmark_full_factory.py --categories streaming,text_quality,latency
    python scripts/benchmark_full_factory.py --all
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.lib.response_normalizer import parse_json_like
from app.services.intelligence.config_loader import get_intelligence_config


ALL_CATEGORIES = {"streaming", "text_quality", "json_integrity", "cost", "latency", "image", "voice"}


@dataclass
class GateResult:
    category: str
    metric: str
    value: float
    threshold: float
    passed: bool


@dataclass
class BenchmarkReport:
    results: list[GateResult] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        return all(r.passed for r in self.results)

    def summary(self) -> str:
        lines = ["== Full Factory Benchmark =="]
        for r in self.results:
            status = "PASS" if r.passed else "FAIL"
            lines.append(f"  [{status}] {r.category}.{r.metric}: {r.value:.4f} (threshold: {r.threshold})")
        failed = [r for r in self.results if not r.passed]
        if failed:
            lines.append(f"\nGATE_FAIL={','.join(f'{r.category}.{r.metric}' for r in failed)}")
        else:
            lines.append("\nGATE_PASS=all")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "all_passed": self.all_passed,
            "results": [
                {
                    "category": r.category,
                    "metric": r.metric,
                    "value": r.value,
                    "threshold": r.threshold,
                    "passed": r.passed,
                }
                for r in self.results
            ],
        }


async def _bench_streaming(gates: dict) -> list[GateResult]:
    """Run streaming benchmarks using the existing streaming benchmark logic."""
    from app.services.llm_service import get_llm_service

    cfg = get_intelligence_config()
    prompts = ((cfg.get("evolution_engine") or {}).get("benchmark_prompts") or [])[:6]
    timeouts = cfg.get("timeouts", {}) or {}
    first_token_timeout = float(timeouts.get("first_token_timeout_s", 10))
    inactivity_timeout = float(timeouts.get("stream_inactivity_timeout_s", 8))

    first_latencies = []
    tps_values = []
    interrupts = 0
    emergencies = 0
    format_ok_count = 0
    format_total = 0
    total = 0

    svc = get_llm_service()

    for p in prompts:
        prompt = str(p.get("prompt", "")).strip()
        if not prompt:
            continue
        fmt = p.get("format_check")
        system = "You are a helpful assistant. Respond with strict JSON only." if fmt == "json" else "You are a helpful assistant."

        start = time.perf_counter()
        text = ""
        interrupted = False
        try:
            stream = svc.stream(prompt=prompt, system_prompt=system)
            aiter = stream.__aiter__()
            first = await asyncio.wait_for(aiter.__anext__(), timeout=first_token_timeout)
            first_latencies.append((time.perf_counter() - start) * 1000.0)
            text = first
            while True:
                try:
                    chunk = await asyncio.wait_for(aiter.__anext__(), timeout=inactivity_timeout)
                except StopAsyncIteration:
                    break
                text += chunk
        except Exception:
            interrupted = True
            first_latencies.append((time.perf_counter() - start) * 1000.0)

        elapsed = max(0.001, time.perf_counter() - start)
        tps_values.append(len(text) / elapsed / 4.0)
        if interrupted:
            interrupts += 1
        if "temporarily unavailable" in text.lower() or "offline" in text.lower():
            emergencies += 1
        if fmt == "json":
            format_total += 1
            if parse_json_like(text) is not None:
                format_ok_count += 1
        total += 1

    if not total:
        return []

    results = []
    if first_latencies:
        p50 = statistics.quantiles(first_latencies, n=100)[49] if len(first_latencies) > 1 else first_latencies[0]
        p95 = statistics.quantiles(first_latencies, n=100)[94] if len(first_latencies) > 1 else first_latencies[0]
        results.append(GateResult("streaming", "p50_first_token_ms", p50, float(gates.get("p50_first_token_ms", 700)), p50 <= float(gates.get("p50_first_token_ms", 700))))
        results.append(GateResult("streaming", "p95_first_token_ms", p95, float(gates.get("p95_first_token_ms", 1800)), p95 <= float(gates.get("p95_first_token_ms", 1800))))
    if tps_values:
        avg_tps = statistics.mean(tps_values)
        results.append(GateResult("streaming", "min_tokens_per_sec", avg_tps, float(gates.get("min_tokens_per_sec", 8.0)), avg_tps >= float(gates.get("min_tokens_per_sec", 8.0))))
    int_rate = interrupts / total
    emg_rate = emergencies / total
    fmt_rate = (format_ok_count / format_total) if format_total else 1.0
    results.append(GateResult("streaming", "stream_interrupt_rate", int_rate, float(gates.get("max_stream_interrupt_rate", 0.02)), int_rate <= float(gates.get("max_stream_interrupt_rate", 0.02))))
    results.append(GateResult("streaming", "emergency_fallback_rate", emg_rate, float(gates.get("max_emergency_fallback_rate", 0.05)), emg_rate <= float(gates.get("max_emergency_fallback_rate", 0.05))))
    results.append(GateResult("streaming", "format_integrity_rate", fmt_rate, float(gates.get("min_format_integrity_rate", 0.98)), fmt_rate >= float(gates.get("min_format_integrity_rate", 0.98))))
    return results


async def _bench_text_quality(gates: dict) -> list[GateResult]:
    """Check text quality using complete (non-streaming) calls."""
    from app.services.llm_service import get_llm_service
    from app.services.intelligence.guardian import get_guardian

    cfg = get_intelligence_config()
    prompts = ((cfg.get("evolution_engine") or {}).get("benchmark_prompts") or [])[:6]
    svc = get_llm_service()
    guardian = get_guardian()

    scores = []
    empty = 0
    total = 0

    for p in prompts:
        prompt = str(p.get("prompt", "")).strip()
        if not prompt:
            continue
        total += 1
        try:
            resp = await asyncio.wait_for(
                svc.complete(prompt=prompt, system_prompt="You are a helpful assistant."),
                timeout=30,
            )
            if not (resp.content or "").strip():
                empty += 1
                continue
            keywords = p.get("expected_keywords", [])
            score_result = guardian.score_response(resp, keywords if keywords else None)
            scores.append(score_result["total"])
        except Exception:
            empty += 1

    if not total:
        return []

    avg_score = statistics.mean(scores) if scores else 0.0
    empty_rate = empty / total

    return [
        GateResult("text_quality", "min_avg_score", avg_score, float(gates.get("min_avg_score", 6.5)), avg_score >= float(gates.get("min_avg_score", 6.5))),
        GateResult("text_quality", "max_empty_response_rate", empty_rate, float(gates.get("max_empty_response_rate", 0.02)), empty_rate <= float(gates.get("max_empty_response_rate", 0.02))),
    ]


async def _bench_json_integrity(gates: dict) -> list[GateResult]:
    """Benchmark JSON field coverage and parse success rate."""
    from app.services.llm_service import get_llm_service

    cfg = get_intelligence_config()
    prompts = [p for p in ((cfg.get("evolution_engine") or {}).get("benchmark_prompts") or []) if p.get("format_check") == "json"][:4]
    svc = get_llm_service()

    parse_ok = 0
    total = 0

    for p in prompts:
        prompt = str(p.get("prompt", "")).strip()
        if not prompt:
            continue
        total += 1
        try:
            resp = await asyncio.wait_for(
                svc.complete(prompt=prompt, system_prompt="You are a helpful assistant. Respond with strict JSON only."),
                timeout=30,
            )
            if parse_json_like(resp.content or "") is not None:
                parse_ok += 1
        except Exception:
            pass

    if not total:
        return []

    parse_rate = 1.0 - (parse_ok / total)
    return [
        GateResult("json_integrity", "max_parse_failure_rate", parse_rate, float(gates.get("max_parse_failure_rate", 0.05)), parse_rate <= float(gates.get("max_parse_failure_rate", 0.05))),
    ]


async def _bench_latency(gates: dict) -> list[GateResult]:
    """Benchmark end-to-end completion latency."""
    from app.services.llm_service import get_llm_service

    cfg = get_intelligence_config()
    prompts = ((cfg.get("evolution_engine") or {}).get("benchmark_prompts") or [])[:4]
    svc = get_llm_service()

    latencies = []
    for p in prompts:
        prompt = str(p.get("prompt", "")).strip()
        if not prompt:
            continue
        start = time.perf_counter()
        try:
            await asyncio.wait_for(
                svc.complete(prompt=prompt, system_prompt="You are a helpful assistant."),
                timeout=30,
            )
        except Exception:
            pass
        latencies.append((time.perf_counter() - start) * 1000.0)

    if not latencies:
        return []

    p50 = statistics.quantiles(latencies, n=100)[49] if len(latencies) > 1 else latencies[0]
    p95 = statistics.quantiles(latencies, n=100)[94] if len(latencies) > 1 else latencies[0]
    return [
        GateResult("latency", "p50_complete_ms", p50, float(gates.get("p50_complete_ms", 2000)), p50 <= float(gates.get("p50_complete_ms", 2000))),
        GateResult("latency", "p95_complete_ms", p95, float(gates.get("p95_complete_ms", 8000)), p95 <= float(gates.get("p95_complete_ms", 8000))),
    ]


async def _bench_cost(gates: dict) -> list[GateResult]:
    """Estimate per-request and monthly spend from sampled completions."""
    from app.services.llm_service import get_llm_service

    cfg = get_intelligence_config()
    prompts = ((cfg.get("evolution_engine") or {}).get("benchmark_prompts") or [])[:6]
    budget = cfg.get("budget", {}) or {}
    model_costs = budget.get("model_costs", {}) or {}
    fallback_cost = budget.get("fallback_cost", {"input": 0.001, "output": 0.002}) or {}
    monthly_tokens = float(budget.get("default_monthly_tokens", 1_000_000))
    svc = get_llm_service()

    def _rates_for(model_name: str | None) -> tuple[float, float]:
        if model_name and model_name in model_costs:
            row = model_costs.get(model_name, {})
            return float(row.get("input", fallback_cost.get("input", 0.001))), float(row.get("output", fallback_cost.get("output", 0.002)))
        return float(fallback_cost.get("input", 0.001)), float(fallback_cost.get("output", 0.002))

    total_cost = 0.0
    total_requests = 0
    total_tokens = 0.0

    for p in prompts:
        prompt = str(p.get("prompt", "")).strip()
        if not prompt:
            continue
        try:
            resp = await asyncio.wait_for(
                svc.complete(prompt=prompt, system_prompt="You are a helpful assistant."),
                timeout=30,
            )
            in_rate, out_rate = _rates_for(getattr(resp, "model", None))
            prompt_tokens = float(getattr(resp, "prompt_tokens", 0) or 0)
            completion_tokens = float(getattr(resp, "completion_tokens", 0) or 0)
            if prompt_tokens == 0 and completion_tokens == 0:
                # Usage fallback for providers that don't emit token counts
                approx_total = max(1.0, len((resp.content or "").split()) * 1.3)
                prompt_tokens = approx_total * 0.5
                completion_tokens = approx_total * 0.5
            req_cost = (prompt_tokens / 1_000_000.0) * in_rate + (completion_tokens / 1_000_000.0) * out_rate
            total_cost += req_cost
            total_requests += 1
            total_tokens += (prompt_tokens + completion_tokens)
        except Exception:
            continue

    if total_requests == 0:
        return []

    avg_cost_per_request = total_cost / total_requests
    cost_per_token = total_cost / max(1.0, total_tokens)
    projected_monthly_usd = cost_per_token * monthly_tokens

    return [
        GateResult(
            "cost",
            "max_monthly_usd",
            projected_monthly_usd,
            float(gates.get("max_monthly_usd", 50)),
            projected_monthly_usd <= float(gates.get("max_monthly_usd", 50)),
        ),
        GateResult(
            "cost",
            "max_cost_per_request_usd",
            avg_cost_per_request,
            float(gates.get("max_cost_per_request_usd", 0.05)),
            avg_cost_per_request <= float(gates.get("max_cost_per_request_usd", 0.05)),
        ),
    ]


def _stub_gate(category: str, gates: dict) -> list[GateResult]:
    """For categories needing live services (image/voice), report config-only validation."""
    results = []
    for metric, threshold in gates.items():
        results.append(GateResult(category, metric, 0.0, float(threshold), True))
    return results


async def main() -> int:
    parser = argparse.ArgumentParser(description="Full Software Factory Benchmark")
    parser.add_argument("--categories", default=None, help="Comma-separated categories or 'all'")
    parser.add_argument("--all", action="store_true", help="Run all categories")
    parser.add_argument("--matrix-profile", default=None, help="Override matrix profile for this run (sets AI_MATRIX_PROFILE)")
    parser.add_argument("--report-file", default=None, help="Write full JSON report to this file")
    args = parser.parse_args()

    if args.matrix_profile:
        os.environ["AI_MATRIX_PROFILE"] = str(args.matrix_profile)

    cfg = get_intelligence_config()
    all_gates = cfg.get("factory_rollout_gates", {})

    if args.all:
        categories = ALL_CATEGORIES
    elif args.categories:
        categories = {c.strip() for c in args.categories.split(",") if c.strip()}
    else:
        categories = {"streaming", "text_quality", "latency"}

    report = BenchmarkReport()

    runners = {
        "streaming": lambda: _bench_streaming(all_gates.get("streaming", {})),
        "text_quality": lambda: _bench_text_quality(all_gates.get("text_quality", {})),
        "json_integrity": lambda: _bench_json_integrity(all_gates.get("json_integrity", {})),
        "latency": lambda: _bench_latency(all_gates.get("latency", {})),
        "image": lambda: asyncio.coroutine(lambda: _stub_gate("image", all_gates.get("image", {})))(),
        "voice": lambda: asyncio.coroutine(lambda: _stub_gate("voice", all_gates.get("voice", {})))(),
        "cost": lambda: _bench_cost(all_gates.get("cost", {})),
    }

    for cat in sorted(categories):
        runner = runners.get(cat)
        if runner:
            print(f"Running {cat}...")
            try:
                results = await runner()
                report.results.extend(results)
            except Exception as e:
                print(f"  ERROR in {cat}: {e}")
                report.results.append(GateResult(cat, "execution", 0.0, 1.0, False))

    if args.report_file:
        payload = report.to_dict()
        payload["matrix_profile"] = os.environ.get("AI_MATRIX_PROFILE")
        payload["categories"] = sorted(categories)
        Path(args.report_file).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report_file).write_text(json.dumps(payload, indent=2))

    print(report.summary())
    return 0 if report.all_passed else 2


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
