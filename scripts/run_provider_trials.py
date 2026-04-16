#!/usr/bin/env python3
"""
Run provider trial comparisons using benchmark_full_factory reports.

This script does not modify config files. It compares baseline profile vs
candidate-first chains and reports ROI gate pass/fail.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.intelligence.config_loader import get_global_driver_chain, get_intelligence_config


def _metric_map(report: dict) -> dict[str, float]:
    out = {}
    for row in report.get("results", []):
        key = f"{row.get('category')}.{row.get('metric')}"
        out[key] = float(row.get("value", 0.0))
    return out


def _run_benchmark(report_path: Path, *, matrix_profile: str, driver_chain_override: str | None = None) -> dict:
    env = os.environ.copy()
    env["AI_MATRIX_PROFILE"] = matrix_profile
    if driver_chain_override:
        env["AI_DRIVER_CHAIN"] = driver_chain_override
    cmd = [
        sys.executable,
        "scripts/benchmark_full_factory.py",
        "--categories",
        "streaming,text_quality,json_integrity,cost,latency",
        "--report-file",
        str(report_path),
    ]
    subprocess.run(cmd, check=False, cwd=str(ROOT), env=env)
    if not report_path.exists():
        return {"all_passed": False, "results": []}
    return json.loads(report_path.read_text())


def main() -> int:
    cfg = get_intelligence_config()
    trials = cfg.get("provider_trials", {}) or {}
    if not trials.get("enabled", False):
        print("provider_trials disabled in intelligence_config.yaml")
        return 0

    compare_profile = str(trials.get("compare_against_profile", "balanced_default"))
    candidates = [str(c).strip() for c in (trials.get("candidates") or []) if str(c).strip()][:2]
    if not candidates:
        print("No provider trial candidates configured.")
        return 0

    reports_dir = ROOT / "reports" / "matrix" / "trials"
    reports_dir.mkdir(parents=True, exist_ok=True)

    baseline_report_path = reports_dir / "baseline.json"
    baseline = _run_benchmark(baseline_report_path, matrix_profile=compare_profile)
    baseline_m = _metric_map(baseline)
    base_chain = get_global_driver_chain()

    gates = (trials.get("benchmark_gates") or {}) if isinstance(trials.get("benchmark_gates"), dict) else {}

    summary = {"baseline_profile": compare_profile, "candidates": []}
    for cand in candidates:
        chain = [cand] + [d for d in base_chain if d != cand]
        report_path = reports_dir / f"{cand}.json"
        report = _run_benchmark(report_path, matrix_profile=compare_profile, driver_chain_override=",".join(chain))
        trial_m = _metric_map(report)

        cost_ok = True
        if gates.get("require_cost_non_increase", True):
            cost_ok = trial_m.get("cost.max_monthly_usd", 0.0) <= baseline_m.get("cost.max_monthly_usd", float("inf"))
        quality_ok = True
        if gates.get("require_text_quality_non_regression", True):
            quality_ok = trial_m.get("text_quality.min_avg_score", 0.0) >= baseline_m.get("text_quality.min_avg_score", 0.0)
        json_ok = True
        if gates.get("require_json_parse_non_regression", True):
            json_ok = trial_m.get("json_integrity.max_parse_failure_rate", 1.0) <= baseline_m.get("json_integrity.max_parse_failure_rate", 1.0)
        stream_ok = trial_m.get("streaming.stream_interrupt_rate", 1.0) <= float(gates.get("max_stream_interrupt_rate", 0.02))

        accepted = bool(report.get("all_passed", False) and cost_ok and quality_ok and json_ok and stream_ok)
        summary["candidates"].append(
            {
                "provider": cand,
                "accepted": accepted,
                "checks": {
                    "factory_all_passed": bool(report.get("all_passed", False)),
                    "cost_non_increase": cost_ok,
                    "text_quality_non_regression": quality_ok,
                    "json_parse_non_regression": json_ok,
                    "stream_interrupt_gate": stream_ok,
                },
                "report_file": str(report_path),
            }
        )

    summary_path = reports_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Provider trial summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
