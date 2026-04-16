#!/usr/bin/env python3
"""
Software-factory streaming benchmark for rollout gates.

Measures:
- first token latency (p50 / p95)
- throughput (tokens-ish chars/sec)
- interruption rate
- emergency fallback rate
- format integrity for JSON-designated prompts
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.lib.response_normalizer import parse_json_like
from app.services.intelligence.config_loader import get_intelligence_config
from app.services.llm_service import get_llm_service


@dataclass
class RunResult:
    first_token_ms: float
    tokens_per_sec: float
    interrupted: bool
    emergency: bool
    format_ok: bool


async def _run_one(
    *,
    prompt: str,
    format_check: str | None,
    driver: str | None,
    model: str | None,
    first_token_timeout_s: float,
    inactivity_timeout_s: float,
) -> RunResult:
    svc = get_llm_service()
    system_prompt = (
        "You are a helpful assistant. Respond with strict JSON only."
        if format_check == "json"
        else "You are a helpful assistant."
    )
    stream = svc.stream(
        prompt=prompt,
        system_prompt=system_prompt,
        driver=driver,
        model=model,
    )
    aiter = stream.__aiter__()
    start = time.perf_counter()
    interrupted = False
    first = ""
    try:
        first = await asyncio.wait_for(aiter.__anext__(), timeout=first_token_timeout_s)
    except Exception:
        interrupted = True
    first_token_ms = (time.perf_counter() - start) * 1000.0
    text = first if first else ""
    chars = len(first)

    if first:
        while True:
            try:
                chunk = await asyncio.wait_for(aiter.__anext__(), timeout=inactivity_timeout_s)
            except StopAsyncIteration:
                break
            except Exception:
                interrupted = True
                break
            text += chunk
            chars += len(chunk)

    elapsed = max(0.001, time.perf_counter() - start)
    tokens_per_sec = chars / elapsed / 4.0  # rough chars->token approximation
    emergency = "temporarily unavailable" in text.lower() or "offline" in text.lower()
    format_ok = True
    if format_check == "json":
        format_ok = parse_json_like(text) is not None

    return RunResult(
        first_token_ms=first_token_ms,
        tokens_per_sec=tokens_per_sec,
        interrupted=interrupted,
        emergency=emergency,
        format_ok=format_ok,
    )


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--driver", default="fast_local")
    parser.add_argument("--model", default=None)
    parser.add_argument("--runs-per-prompt", type=int, default=1)
    args = parser.parse_args()

    cfg = get_intelligence_config()
    gates = ((cfg.get("factory_rollout_gates") or {}).get("streaming") or {})
    prompts = ((cfg.get("evolution_engine") or {}).get("benchmark_prompts") or [])[:10]
    timeouts = cfg.get("timeouts", {}) or {}
    first_token_timeout_s = float(timeouts.get("first_token_timeout_s", 10))
    inactivity_timeout_s = float(timeouts.get("stream_inactivity_timeout_s", 8))

    results: list[RunResult] = []
    for p in prompts:
        prompt = str(p.get("prompt", "")).strip()
        format_check = p.get("format_check")
        if not prompt:
            continue
        for _ in range(max(1, args.runs_per_prompt)):
            r = await _run_one(
                prompt=prompt,
                format_check=format_check,
                driver=args.driver or None,
                model=args.model,
                first_token_timeout_s=first_token_timeout_s,
                inactivity_timeout_s=inactivity_timeout_s,
            )
            results.append(r)

    if not results:
        print("No prompts to benchmark.")
        return 1

    first_latencies = [r.first_token_ms for r in results]
    p50 = statistics.quantiles(first_latencies, n=100)[49] if len(first_latencies) > 1 else first_latencies[0]
    p95 = statistics.quantiles(first_latencies, n=100)[94] if len(first_latencies) > 1 else first_latencies[0]
    tps = statistics.mean(r.tokens_per_sec for r in results)
    interruption_rate = sum(1 for r in results if r.interrupted) / len(results)
    emergency_rate = sum(1 for r in results if r.emergency) / len(results)
    format_rate = sum(1 for r in results if r.format_ok) / len(results)

    print("== Streaming Benchmark ==")
    print(f"runs={len(results)} driver={args.driver} model={args.model or 'default'}")
    print(f"p50_first_token_ms={p50:.2f}")
    print(f"p95_first_token_ms={p95:.2f}")
    print(f"tokens_per_sec={tps:.2f}")
    print(f"stream_interrupt_rate={interruption_rate:.4f}")
    print(f"emergency_fallback_rate={emergency_rate:.4f}")
    print(f"format_integrity_rate={format_rate:.4f}")

    checks = [
        ("p50_first_token_ms", p50 <= float(gates.get("p50_first_token_ms", 700))),
        ("p95_first_token_ms", p95 <= float(gates.get("p95_first_token_ms", 1800))),
        ("tokens_per_sec", tps >= float(gates.get("min_tokens_per_sec", 8.0))),
        (
            "stream_interrupt_rate",
            interruption_rate <= float(gates.get("max_stream_interrupt_rate", 0.02)),
        ),
        (
            "emergency_fallback_rate",
            emergency_rate <= float(gates.get("max_emergency_fallback_rate", 0.05)),
        ),
        (
            "format_integrity_rate",
            format_rate >= float(gates.get("min_format_integrity_rate", 0.98)),
        ),
    ]
    failed = [name for name, ok in checks if not ok]
    if failed:
        print(f"GATE_FAIL={','.join(failed)}")
        return 2
    print("GATE_PASS=all")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
