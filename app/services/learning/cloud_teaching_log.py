"""Append-only JSONL for cloud → local distillation datasets.

Used by Brain._auto_train, streaming fallback, and hybrid cloud-wins paths.
Configure via intelligence_config.yaml → learning.cloud_teaching.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from app.services.intelligence.config_loader import get_intelligence_config

logger = logging.getLogger(__name__)

_LOCAL_DRIVERS = frozenset({"ollama", "bitnet", "fast_local"})


def append_cloud_teaching_record(
    agent_type: str,
    prompt: str,
    response: str,
    *,
    source_driver: str,
    model: str | None = None,
    quality_score: float | None = None,
) -> None:
    """Append one NDJSON line to training/cloud_teaching/{agent}.jsonl.

    Skips local-only drivers. When quality_score is None, min_quality_score
    is not applied (streaming / unknown quality). When set, rows below
    min_quality_score are dropped to avoid poisoning fine-tune data.
    """
    if not response or not str(response).strip():
        return
    drv = (source_driver or "").strip().lower()
    if drv in _LOCAL_DRIVERS or drv in ("", "unknown"):
        return

    cfg = (get_intelligence_config().get("learning") or {}).get("cloud_teaching") or {}
    if not cfg.get("enabled", True):
        return

    min_q = float(cfg.get("min_quality_score", 0.0))
    if quality_score is not None and quality_score < min_q:
        return

    max_p = int(cfg.get("max_prompt_chars", 200_000))
    max_r = int(cfg.get("max_response_chars", 200_000))
    p = prompt if len(prompt) <= max_p else prompt[:max_p]
    r = response if len(response) <= max_r else response[:max_r]

    training_dir = Path(cfg.get("output_dir", "training/cloud_teaching"))
    training_dir.mkdir(parents=True, exist_ok=True)
    log_path = training_dir / f"{agent_type}.jsonl"
    entry = {
        "agent": agent_type,
        "prompt": p,
        "response": r,
        "source": source_driver,
        "model": model or "",
    }
    if quality_score is not None:
        entry["quality_score"] = round(float(quality_score), 3)

    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError as e:
        logger.debug("cloud_teaching append skipped: %s", e)
