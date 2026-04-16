#!/usr/bin/env python3
"""
Verify NVIDIA NIM / integrate API: list available models and compare to config.

Uses GET {base_url}/models (OpenAI-compatible). Requires NVIDIA_API_KEY in env.

Does not print secrets. Exit 0 if all configured chat models are found (or no key
with --dry-run to only show configured IDs).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import httpx
import yaml

from app.config import get_settings
from app.services.intelligence.config_loader import get_provider_config


def _collect_configured_models() -> tuple[str, list[str]]:
    """(default_model, all_unique_ids_from_settings_and_yaml)."""
    s = get_settings()
    out: list[str] = []
    if s.nvidia_model:
        out.append(s.nvidia_model.strip())

    from app.services.intelligence import config_loader as ic

    cfg = ic.get_intelligence_config()
    prov = (cfg.get("providers") or {}).get("nvidia", {}) or {}
    for m in prov.get("fallback_models") or []:
        if m:
            out.append(str(m).strip())
    if prov.get("fallback_model"):
        out.append(str(prov["fallback_model"]).strip())

    tiers = ((cfg.get("smart_router") or {}).get("model_tiers") or {}).get("nvidia") or {}
    for v in tiers.values():
        if v:
            out.append(str(v).strip())

    voice = ((cfg.get("voice_models") or {}).get("nvidia") or {}).get("stt")
    if voice:
        out.append(str(voice).strip())

    seen: set[str] = set()
    uniq: list[str] = []
    for m in out:
        if m and m not in seen:
            seen.add(m)
            uniq.append(m)
    return (s.nvidia_model or "", uniq)


def _fetch_remote_model_ids(base_url: str, api_key: str, timeout: float = 30.0) -> list[str]:
    url = base_url.rstrip("/") + "/models"
    headers = {"Authorization": f"Bearer {api_key}", "Accept": "application/json"}
    with httpx.Client(timeout=timeout) as client:
        r = client.get(url, headers=headers)
        r.raise_for_status()
        data = r.json()
    items = data.get("data") or data.get("models") or []
    ids: list[str] = []
    for it in items:
        if isinstance(it, dict) and it.get("id"):
            ids.append(str(it["id"]))
        elif isinstance(it, str):
            ids.append(it)
    return ids


def _normalize(s: str) -> str:
    return s.strip().lower()


def model_available(configured: str, remote_ids: list[str]) -> bool:
    """True if configured id matches any remote id (exact or suffix)."""
    c = configured.strip()
    if not c:
        return False
    remote_set = {_normalize(x) for x in remote_ids}
    cn = _normalize(c)
    if cn in remote_set:
        return True
    # Some catalogs return without vendor prefix
    for rid in remote_ids:
        rn = _normalize(rid)
        if cn == rn or cn.endswith(rn) or rn.endswith(cn):
            return True
        if "/" in c:
            short = c.split("/")[-1]
            if _normalize(short) == rn or rn.endswith(_normalize(short)):
                return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print configured model IDs; do not call NVIDIA API",
    )
    args = parser.parse_args()

    default_m, configured = _collect_configured_models()
    prov = get_provider_config("nvidia") or {}
    base_url = str(prov.get("base_url") or "https://integrate.api.nvidia.com/v1")

    print("NVIDIA model check")
    print(f"  base_url: {base_url}")
    print(f"  default (settings.nvidia_model): {default_m or '(empty)'}")
    print(f"  configured IDs ({len(configured)}): {', '.join(configured)}")

    s = get_settings()
    key = (s.nvidia_api_key or "").strip()
    if not key:
        print("\n  NVIDIA_API_KEY is not set — cannot query /v1/models.")
        print("  Set the key in .env then re-run without --dry-run.")
        return 1

    if args.dry_run:
        return 0

    try:
        remote = _fetch_remote_model_ids(base_url, key)
    except httpx.HTTPStatusError as e:
        print(f"\n  HTTP error listing models: {e.response.status_code}")
        try:
            print(f"  body: {e.response.text[:500]}")
        except Exception:
            pass
        return 2
    except Exception as e:
        print(f"\n  Failed to list models: {e}")
        return 2

    print(f"\n  Remote models returned: {len(remote)}")
    # Optional: write full list to stdout truncated
    if len(remote) <= 30:
        print(f"  ids: {', '.join(remote)}")
    else:
        print(f"  sample: {', '.join(remote[:15])} ... ({len(remote) - 15} more)")

    missing: list[str] = []
    for mid in configured:
        if not model_available(mid, remote):
            missing.append(mid)

    if missing:
        print("\n  NOT FOUND on NVIDIA API (may be renamed, gated, or different product):")
        for m in missing:
            print(f"    - {m}")
        print(
            "\n  Note: speech models (e.g. parakeet-*) may live under Riva / NIM speech "
            "catalogs, not chat /v1/models."
        )
        return 3

    print("\n  All configured chat-related model IDs appear in the API catalog.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
