#!/usr/bin/env python3
"""
Freeze current model-matrix baseline into a JSON artifact.

Usage:
  python scripts/model_matrix_baseline.py
  python scripts/model_matrix_baseline.py --profile budget_strict --out reports/matrix/baseline_budget.json
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys
import yaml

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import get_settings


def _json_ready(obj):
    if isinstance(obj, dict):
        return {str(k): _json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [_json_ready(v) for v in obj]
    return obj


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze model matrix baseline snapshot")
    parser.add_argument("--profile", default=None, help="Matrix profile name override")
    parser.add_argument("--out", default="reports/matrix/baseline_snapshot.json", help="Output JSON path")
    args = parser.parse_args()

    if args.profile:
        os.environ["AI_MATRIX_PROFILE"] = str(args.profile)

    s = get_settings()
    cfg_path = ROOT / "intelligence_config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else {}
    matrix_root = cfg.get("model_matrix", {}) or {}
    profiles = matrix_root.get("profiles", {}) if isinstance(matrix_root.get("profiles"), dict) else {}
    active_profile = str(s.ai_matrix_profile or matrix_root.get("active_profile") or "balanced_default")
    matrix = profiles.get(active_profile, {}) if isinstance(profiles.get(active_profile, {}), dict) else {}
    chain = matrix.get("global_driver_chain") or ((cfg.get("resilience") or {}).get("global_driver_chain") or [])

    key_map = {
        "openai": bool(s.openai_api_key),
        "anthropic": bool(s.anthropic_api_key),
        "gemini": bool(s.gemini_api_key),
        "groq": bool(s.groq_api_key),
        "sarvam": bool(s.sarvam_api_key),
        "nvidia": bool(s.nvidia_api_key),
        "together": bool(getattr(s, "together_api_key", "")),
        "fireworks": bool(getattr(s, "fireworks_api_key", "")),
        "bitnet": True,
        "ollama": True,
        "fast_local": bool(((cfg.get("providers") or {}).get("fast_local") or {}).get("base_url")),
    }
    configured = [{"driver": str(name), "configured": bool(key_map.get(str(name), False))} for name in chain]

    out = {
        "generated_at": int(time.time()),
        "ai_matrix_profile": active_profile,
        "global_driver_chain": chain,
        "configured_drivers": configured,
        "smart_router_driver_chains": (cfg.get("smart_router") or {}).get("driver_chains", {}),
        "smart_router_model_tiers": (cfg.get("smart_router") or {}).get("model_tiers", {}),
        "matrix_profile_config": matrix,
        "budget": cfg.get("budget", {}),
        "factory_rollout_gates": cfg.get("factory_rollout_gates", {}),
        "provider_trials": cfg.get("provider_trials", {}),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(_json_ready(out), indent=2))
    print(f"Baseline snapshot written: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
