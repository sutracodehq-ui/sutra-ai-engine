"""
Platform status — DB, Redis, LLM endpoints, Qdrant, limits, host metrics.

Used by /health/full and /metrics. No secrets in responses (only booleans / counts).
"""

from __future__ import annotations

import asyncio
import logging
import os
import resource
import sys
import time
from typing import Any

import httpx
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.services.intelligence.config_loader import (
    get_global_driver_chain,
    get_intelligence_config,
    get_local_driver_ids,
    get_provider_config,
)
from app.services.intelligence.driver import get_driver_registry

logger = logging.getLogger(__name__)

_HTTP_TIMEOUT = 3.0


def _process_memory_mb() -> float | None:
    """Approximate process RSS in MiB (platform-specific ru_maxrss units)."""
    try:
        rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if sys.platform == "darwin":
            return round(rss / (1024 * 1024), 2)
        return round(rss / 1024.0, 2)
    except Exception:
        return None


def collect_system_metrics() -> dict[str, Any]:
    """CPU, load, memory (process), disk not included (optional psutil not in deps)."""
    out: dict[str, Any] = {
        "cpu_count": os.cpu_count(),
        "timestamp_unix": time.time(),
    }
    try:
        out["load_average"] = list(os.getloadavg())
    except OSError:
        out["load_average"] = None
    mem = _process_memory_mb()
    if mem is not None:
        out["process_rss_mib"] = mem
    return out


def collect_limits_and_config() -> dict[str, Any]:
    """Admission, VPS profiles, rate limits, LLM parallelism — from settings + YAML."""
    s = get_settings()
    intel = get_intelligence_config()
    res = intel.get("resilience") or {}
    admission = res.get("admission_control") or {}
    vps = intel.get("vps_admission") or {}
    rl = res.get("rate_limiter") or {}
    budget = intel.get("budget") or {}
    timeouts = intel.get("timeouts") or {}
    return {
        "llm_max_parallel": s.llm_max_parallel,
        "admission_control": dict(admission),
        "vps_admission": {"enabled": vps.get("enabled", False), "profiles": list((vps.get("profiles") or {}).keys())},
        "rate_limiter_default_rpm": rl.get("default_rpm"),
        "budget_default_monthly_tokens": budget.get("default_monthly_tokens"),
        "timeouts_first_token_s": timeouts.get("first_token_timeout_s")
        or (res.get("timeouts") or {}).get("first_token_timeout_s"),
        "timeouts_stream_inactivity_s": timeouts.get("stream_inactivity_timeout_s")
        or (res.get("timeouts") or {}).get("stream_inactivity_timeout_s"),
    }


def collect_driver_and_circuit_snapshot() -> dict[str, Any]:
    """Which drivers are constructible + circuit breaker states + brain queue."""
    out: dict[str, Any] = {"drivers": {}, "circuit_breaker": {}, "brain_queue": {}}
    try:
        reg = get_driver_registry()
        names = [n for n in get_global_driver_chain() if n]
        if not names:
            out["drivers"]["_note"] = "resilience.global_driver_chain is empty — no drivers enumerated"
        for name in names:
            try:
                out["drivers"][name] = {"configured": reg._driver_configured(name)}
            except Exception as e:
                out["drivers"][name] = {"configured": False, "error": str(e)[:120]}
    except Exception as e:
        out["drivers"]["_error"] = str(e)[:200]

    try:
        from app.services.intelligence.guardian import get_guardian

        out["circuit_breaker"] = get_guardian().circuit_breaker.status()
    except Exception as e:
        out["circuit_breaker"] = {"error": str(e)[:200]}

    try:
        from app.services.intelligence.brain import get_brain

        out["brain_queue"] = get_brain().queue.stats
    except Exception as e:
        out["brain_queue"] = {"error": str(e)[:200]}

    try:
        reg = get_driver_registry()
        reg._ensure_local_stream_semaphore()
        sem = reg._local_stream_sem
        slots = reg._local_stream_slots
        if sem is not None and slots:
            in_use = max(0, slots - sem._value)
            out["local_stream_admission"] = {"max_slots": slots, "available_slots": sem._value, "approx_in_use": in_use}
        else:
            out["local_stream_admission"] = {"max_slots": slots or None, "note": "semaphore not initialized yet"}
    except Exception as e:
        out["local_stream_admission"] = {"error": str(e)[:120]}

    return out


def collect_api_key_flags() -> dict[str, bool]:
    """Which provider env keys are non-empty (no values)."""
    s = get_settings()
    return {
        "openai": bool(s.openai_api_key),
        "anthropic": bool(s.anthropic_api_key),
        "gemini": bool(s.gemini_api_key),
        "groq": bool(s.groq_api_key),
        "nvidia": bool(s.nvidia_api_key),
        "sarvam": bool(s.sarvam_api_key),
        "together": bool(getattr(s, "together_api_key", "")),
        "fireworks": bool(getattr(s, "fireworks_api_key", "")),
        "fal": bool(s.fal_key),
        "elevenlabs": bool(s.elevenlabs_api_key),
        "tavily": bool(s.tavily_api_key),
    }


def _chain_key_diagnostics() -> dict[str, Any]:
    """Explain when fallback chain effectively starts local due missing cloud keys."""
    key_flags = collect_api_key_flags()
    reg = get_driver_registry()
    raw_chain = [n for n in get_global_driver_chain() if n]
    configured_chain = [n for n in raw_chain if reg._driver_configured(n)]
    local_ids = set(get_local_driver_ids())
    cloud_chain = [n for n in raw_chain if n not in local_ids]
    missing_cloud = [n for n in cloud_chain if not reg._driver_configured(n)]
    first_configured = configured_chain[0] if configured_chain else None
    starts_local = bool(first_configured and first_configured in local_ids)
    return {
        "raw_chain": raw_chain,
        "configured_chain": configured_chain,
        "first_configured_driver": first_configured,
        "starts_local": starts_local,
        "missing_cloud_drivers": missing_cloud,
        "api_keys_present": {
            "groq": key_flags.get("groq", False),
            "nvidia": key_flags.get("nvidia", False),
            "openai": key_flags.get("openai", False),
            "anthropic": key_flags.get("anthropic", False),
            "gemini": key_flags.get("gemini", False),
            "sarvam": key_flags.get("sarvam", False),
            "together": key_flags.get("together", False),
            "fireworks": key_flags.get("fireworks", False),
        },
    }


async def _http_json(method: str, url: str, **kwargs) -> tuple[int, Any | None, str | None]:
    try:
        async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
            r = await client.request(method, url, **kwargs)
            body = None
            try:
                body = r.json()
            except Exception:
                body = None
            return r.status_code, body, None if r.is_success else (r.text or "")[:200]
    except Exception as e:
        return 0, None, str(e)[:200]


async def probe_qdrant() -> dict[str, Any]:
    s = get_settings()
    base = (s.qdrant_url or "").rstrip("/")
    if not base:
        return {"status": "skipped", "reason": "no qdrant_url"}
    code, body, err = await _http_json("GET", f"{base}/collections")
    if code == 200:
        return {"status": "ok", "endpoint": "/collections", "collections": isinstance(body, dict)}
    code2, body2, err2 = await _http_json("GET", f"{base}/")
    return {
        "status": "ok" if code2 == 200 else "error",
        "http_status": code2,
        "error": err2 if code2 != 200 else err,
        "body": body2 if code2 == 200 else body,
    }


def _ollama_tag_satisfies(required: str, loaded_name: str) -> bool:
    """
    True if loaded Ollama tag satisfies a configured requirement.

    Handles: exact match, same repo with :latest/:main, and repo-only match
    (e.g. required llama3:8b vs loaded llama3:latest — same model family present).
    """
    w = (required or "").strip()
    n = (loaded_name or "").strip()
    if not w or not n:
        return False
    if w == n:
        return True
    if len(n) > len(w) and n.startswith(w) and n[len(w)] in ":-_":
        return True
    w_parts = w.split(":", 1)
    n_parts = n.split(":", 1)
    w_repo, w_tag = (w_parts[0], w_parts[1] if len(w_parts) > 1 else "")
    n_repo, n_tag = (n_parts[0], n_parts[1] if len(n_parts) > 1 else "")
    if w_repo != n_repo:
        return False
    if not w_tag or not n_tag:
        return True
    if w_tag == n_tag:
        return True
    if n_tag in ("latest", "main") or w_tag in ("latest", "main"):
        return True
    if w_tag in n_tag or n_tag in w_tag:
        return True
    if w in n or n in w:
        return True
    return False


async def probe_ollama_models() -> dict[str, Any]:
    s = get_settings()
    meta = get_provider_config("ollama") or {}
    base = (meta.get("base_url") or s.ollama_base_url or "").rstrip("/")
    if not base:
        return {"status": "skipped", "reason": "no ollama base_url"}
    code, body, err = await _http_json("GET", f"{base}/api/tags")
    if code != 200 or not isinstance(body, dict):
        return {"status": "error", "http_status": code, "error": err, "base_url": base}
    names = [m.get("name") for m in body.get("models", []) if isinstance(m, dict) and m.get("name")]
    want = {s.ollama_model}
    prov = get_provider_config("ollama") or {}
    for m in prov.get("fallback_models") or []:
        if m:
            want.add(str(m).strip())
    if prov.get("fallback_model"):
        want.add(str(prov["fallback_model"]).strip())
    want.discard(None)
    want.discard("")
    missing = [w for w in sorted(want) if not any(_ollama_tag_satisfies(w, n) for n in names)]
    return {
        "status": "ok" if not missing else "degraded",
        "base_url": base,
        "models_loaded": names[:40],
        "models_loaded_truncated": len(names) > 40,
        "models_loaded_count": len(names),
        "required_in_config": sorted(want),
        "missing_models": missing,
    }


async def probe_openai_compatible(driver: str) -> dict[str, Any]:
    s = get_settings()
    if driver == "bitnet":
        api_key = "local"
    elif driver == "fast_local":
        api_key = s.fast_local_api_key or "local"
    else:
        key_attr = f"{driver}_api_key"
        api_key = getattr(s, key_attr, "") or ""
    if not api_key:
        return {"status": "skipped", "reason": "no api key"}
    meta = get_provider_config(driver) or {}
    base = (meta.get("base_url") or "").rstrip("/")
    if not base:
        return {"status": "skipped", "reason": "no base_url in yaml"}
    headers = {"Authorization": f"Bearer {api_key}", "Accept": "application/json"}
    code, body, err = await _http_json("GET", f"{base}/models", headers=headers)
    ok = code == 200 and isinstance(body, dict)
    nmodels = len(body.get("data", [])) if ok else 0
    if driver == "bitnet" and code == 404:
        return {
            "status": "ok",
            "http_status": code,
            "models_reported": 0,
            "note": "OpenAI /v1/models not exposed; server reachable (bitnet.cpp often has no catalog endpoint).",
            "error": None,
        }
    if driver == "fast_local" and code == 0:
        return {
            "status": "skipped",
            "http_status": code,
            "models_reported": 0,
            "reason": "unreachable_or_dns",
            "error": err,
        }
    return {
        "status": "ok" if ok else "error",
        "http_status": code,
        "models_reported": nmodels,
        "error": err if not ok else None,
    }


async def probe_gemini() -> dict[str, Any]:
    s = get_settings()
    if not s.gemini_api_key:
        return {"status": "skipped", "reason": "no api key"}
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={s.gemini_api_key}"
    code, body, err = await _http_json("GET", url)
    ok = code == 200 and isinstance(body, dict)
    n = len(body.get("models", [])) if ok else 0
    return {"status": "ok" if ok else "error", "http_status": code, "models_reported": n, "error": err if not ok else None}


async def probe_llm_driver(name: str) -> dict[str, Any]:
    """Dispatch health probe by driver name (order comes from resilience.global_driver_chain)."""
    if name == "ollama":
        return await probe_ollama_models()
    if name == "gemini":
        return await probe_gemini()
    if name == "anthropic":
        return await probe_anthropic()
    if name in ("groq", "openai", "nvidia", "sarvam", "together", "fireworks", "bitnet", "fast_local"):
        return await probe_openai_compatible(name)
    return {"status": "skipped", "reason": f"no probe for driver {name!r}"}


async def probe_anthropic() -> dict[str, Any]:
    s = get_settings()
    if not s.anthropic_api_key:
        return {"status": "skipped", "reason": "no api key"}
    headers = {
        "x-api-key": s.anthropic_api_key,
        "anthropic-version": "2023-06-01",
        "Accept": "application/json",
    }
    code, body, err = await _http_json("GET", "https://api.anthropic.com/v1/models", headers=headers)
    ok = code == 200
    n = len((body or {}).get("data", [])) if isinstance(body, dict) else 0
    if not ok and code == 404:
        return {"status": "ok", "note": "models list endpoint unavailable; key format accepted", "http_status": code}
    return {"status": "ok" if ok else "error", "http_status": code, "models_reported": n, "error": err if not ok else None}


async def probe_database(db: AsyncSession) -> dict[str, Any]:
    try:
        await db.execute(text("SELECT 1"))
        return {"status": "ok"}
    except Exception as e:
        return {"status": "error", "error": str(e)[:200]}


async def probe_redis(redis) -> dict[str, Any]:
    try:
        pong = await redis.ping()
        ok = pong is True or pong is 1 or str(pong).upper() == "TRUE"
        return {"status": "ok" if ok else "error", "ping": bool(ok)}
    except Exception as e:
        return {"status": "error", "error": str(e)[:200]}


async def probe_migrations(db: AsyncSession) -> dict[str, Any]:
    try:
        result = await db.execute(text("SELECT version_num FROM alembic_version LIMIT 1"))
        row = result.scalar_one_or_none()
        if row:
            return {"status": "ok", "version": row}
        return {"status": "error", "error": "no alembic version row"}
    except Exception as e:
        return {"status": "error", "error": str(e)[:200]}


async def probe_tenants(db: AsyncSession) -> dict[str, Any]:
    try:
        result = await db.execute(text("SELECT COUNT(*) FROM tenants WHERE is_active = true"))
        count = int(result.scalar_one())
        return {"status": "ok", "active_tenants": count}
    except Exception as e:
        return {"status": "error", "error": str(e)[:200]}


async def _run_db_probes_sequential(db: AsyncSession) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """One AsyncSession cannot run concurrent operations — run DB checks in order."""
    db_r = await probe_database(db)
    mig_r = await probe_migrations(db)
    ten_r = await probe_tenants(db)
    return db_r, mig_r, ten_r


async def build_full_status(db: AsyncSession, redis) -> dict[str, Any]:
    """Run core probes + parallel remote probes."""
    t0 = time.perf_counter()

    db_r, mig_r, ten_r = await _run_db_probes_sequential(db)
    redis_r = await probe_redis(redis)

    chain = [n for n in get_global_driver_chain() if n]
    if not chain:
        logger.warning("resilience.global_driver_chain empty — probing default LLM set for /health/full")
        chain = ["groq", "nvidia", "gemini", "together", "fireworks", "openai", "anthropic", "sarvam", "ollama", "bitnet", "fast_local"]

    remote_tasks = await asyncio.gather(
        probe_qdrant(),
        *[probe_llm_driver(n) for n in chain],
        return_exceptions=True,
    )

    names = ("qdrant",) + tuple(chain)
    models: dict[str, Any] = {}
    for i, name in enumerate(names):
        r = remote_tasks[i]
        models[name] = r if not isinstance(r, Exception) else {"status": "error", "error": str(r)[:200]}

    critical_ok = (
        db_r.get("status") == "ok"
        and redis_r.get("status") == "ok"
        and mig_r.get("status") == "ok"
    )

    notices: list[str] = []
    overall = "ok"
    if not critical_ok:
        overall = "unhealthy"
    elif ten_r.get("status") != "ok":
        overall = "degraded"
        notices.append("Tenants probe failed — see components.tenants.")
    elif ten_r.get("status") == "ok" and ten_r.get("active_tenants", 1) == 0:
        overall = "degraded"
        notices.append("No active tenants — seed a tenant for full readiness.")

    ollama_st = models.get("ollama", {}).get("status")
    if ollama_st == "degraded":
        notices.append(
            "Ollama: not all models listed in config are pulled on this host "
            "(see components.models_and_endpoints.ollama.missing_models). "
            "The app can still answer via Groq/Gemini/etc. fallbacks."
        )
    elif ollama_st == "error":
        notices.append(
            "Ollama: /api/tags failed — local inference may be unavailable until Ollama is healthy."
        )
    if models.get("fast_local", {}).get("status") == "skipped":
        notices.append(
            "fast_local: optional vLLM/TGI URL not reachable (DNS or service down)."
        )
    chain_diag = _chain_key_diagnostics()
    if chain_diag.get("starts_local") and chain_diag.get("missing_cloud_drivers"):
        missing = ", ".join(chain_diag["missing_cloud_drivers"])
        notices.append(
            "Configured fallback chain currently starts local because cloud drivers are not configured: "
            f"{missing}. Set API keys so cloud fallback happens before Ollama under load."
        )

    return {
        "status": overall,
        "notices": notices,
        "elapsed_ms": round((time.perf_counter() - t0) * 1000, 2),
        "components": {
            "database": db_r,
            "redis": redis_r,
            "migrations": mig_r,
            "tenants": ten_r,
            "models_and_endpoints": models,
        },
        "system": collect_system_metrics(),
        "limits": collect_limits_and_config(),
        "drivers": collect_driver_and_circuit_snapshot(),
        "api_keys_present": collect_api_key_flags(),
        "chain_diagnostics": chain_diag,
    }


async def build_metrics_snapshot(db: AsyncSession, redis) -> dict[str, Any]:
    """Prometheus-friendly JSON snapshot for /metrics."""
    full = await build_full_status(db, redis)
    return {
        "timestamp_unix": time.time(),
        "status": full.get("status"),
        "notices": full.get("notices", []),
        "elapsed_ms": full.get("elapsed_ms"),
        "system": full.get("system"),
        "limits": full.get("limits"),
        "api_keys_present": full.get("api_keys_present"),
        "drivers": full.get("drivers"),
        "components": full.get("components"),
    }
