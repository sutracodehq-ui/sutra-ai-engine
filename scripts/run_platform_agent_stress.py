#!/usr/bin/env python3
"""
Platform + Agents stress runner (CI-style, resumable).

Phases:
  0) preflight health checks
  1) inventory + smoke (1 happy case per agent)
  2) full matrix (happy + edge + failure cases)
  3) core endpoint checks
  4) stress slices (low/mod/high concurrency)
  5) summary artifacts
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import random
import statistics
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_URL = "http://localhost:8090"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    if not path.exists():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_json_loads(raw: str) -> Any | None:
    try:
        return json.loads(raw)
    except Exception:
        return None


@dataclass
class RunnerConfig:
    base_url: str
    api_key: str
    artifacts_root: Path
    run_id: str
    resume: bool
    max_agents: int
    batch_size: int
    agent_concurrency: int
    request_timeout_s: int
    poll_active_s: int
    poll_idle_s: int
    pause_error_rate: float
    pause_seconds: int
    include_expensive: bool
    stress_levels: list[int]
    stress_rounds: int
    stress_agent_count: int
    profile: str


class ArtifactStore:
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.checkpoint_path = self.base_dir / "checkpoint.json"

    def write_json(self, rel: str, payload: dict | list) -> None:
        p = self.base_dir / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(payload, indent=2))

    def append_jsonl(self, rel: str, row: dict) -> None:
        p = self.base_dir / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(row, ensure_ascii=True)
        with self._lock:
            with p.open("a", encoding="utf-8") as f:
                f.write(line + "\n")

    def load_checkpoint(self) -> dict:
        if self.checkpoint_path.exists():
            data = safe_json_loads(self.checkpoint_path.read_text()) or {}
            data.setdefault("completed_phases", [])
            data.setdefault("completed_case_ids", [])
            data.setdefault("created_at", now_iso())
            return data
        return {
            "created_at": now_iso(),
            "updated_at": now_iso(),
            "completed_phases": [],
            "completed_case_ids": [],
            "notes": [],
        }

    def save_checkpoint(self, checkpoint: dict) -> None:
        checkpoint["updated_at"] = now_iso()
        self.checkpoint_path.write_text(json.dumps(checkpoint, indent=2))


class HttpClient:
    def __init__(self, base_url: str, api_key: str, timeout_s: int):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_s = timeout_s

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def request(
        self,
        method: str,
        path: str,
        json_body: dict | None = None,
        query: dict | None = None,
        timeout_s: int | None = None,
        stream_hint: bool = False,
    ) -> dict[str, Any]:
        t0 = time.perf_counter()
        timeout = timeout_s or self.timeout_s
        url = f"{self.base_url}{path}"
        if query:
            url = f"{url}?{urllib.parse.urlencode(query, doseq=True)}"
        data = None
        if json_body is not None:
            data = json.dumps(json_body).encode("utf-8")

        req = urllib.request.Request(url=url, data=data, method=method.upper(), headers=self._headers())
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
                parsed = safe_json_loads(raw)
                ms = (time.perf_counter() - t0) * 1000
                if stream_hint and not raw.startswith("data:"):
                    # Some stream handlers still emit SSE text. Keep raw for diagnosis.
                    pass
                return {
                    "ok": 200 <= resp.status < 300,
                    "status": resp.status,
                    "ms": round(ms, 2),
                    "json": parsed,
                    "raw": raw[:3000],
                    "error": None,
                }
        except urllib.error.HTTPError as e:
            try:
                body = e.read().decode("utf-8", errors="replace")
            except Exception:
                body = str(e)
            ms = (time.perf_counter() - t0) * 1000
            return {
                "ok": False,
                "status": e.code,
                "ms": round(ms, 2),
                "json": safe_json_loads(body),
                "raw": body[:3000],
                "error": body[:500],
            }
        except Exception as e:
            ms = (time.perf_counter() - t0) * 1000
            return {
                "ok": False,
                "status": 0,
                "ms": round(ms, 2),
                "json": None,
                "raw": "",
                "error": str(e)[:500],
            }


def build_case_matrix(agents: list[dict]) -> list[dict]:
    cases: list[dict] = []
    for agent in agents:
        identifier = str(agent.get("identifier", "")).strip()
        name = str(agent.get("name", identifier))
        domain = str(agent.get("domain", "general"))
        capabilities = agent.get("capabilities") or []
        cap_hint = ", ".join(str(c) for c in capabilities[:3]) if isinstance(capabilities, list) else ""

        happy = (
            f"You are acting as {name} in domain {domain}. "
            f"Give a practical, concise response on improving conversion for a new product launch. "
            f"Use your strengths: {cap_hint or 'analysis and actionable recommendations'}."
        )
        edge = (
            "Return a compact JSON object with keys summary, actions, risks. "
            "Topic: launch week planning for a small team. Keep each value <= 20 words."
        )
        failure_prompt = "X" * 12050  # Intentionally violates max_length guard

        cases.append(
            {
                "case_id": f"{identifier}::happy",
                "agent": identifier,
                "case_type": "happy",
                "prompt": happy,
                "expect_success": True,
                "options": {"max_tokens": 400, "temperature": 0.2},
            }
        )
        cases.append(
            {
                "case_id": f"{identifier}::edge",
                "agent": identifier,
                "case_type": "edge",
                "prompt": edge,
                "expect_success": True,
                "options": {"max_tokens": 350, "temperature": 0.1},
            }
        )
        cases.append(
            {
                "case_id": f"{identifier}::failure",
                "agent": identifier,
                "case_type": "failure",
                "prompt": failure_prompt,
                "expect_success": False,
                "options": {},
            }
        )
    return cases


def percentile(values: list[float], p: int) -> float | None:
    if not values:
        return None
    vals = sorted(values)
    idx = int(round((p / 100) * (len(vals) - 1)))
    idx = max(0, min(len(vals) - 1, idx))
    return round(vals[idx], 2)


def run() -> int:
    parser = argparse.ArgumentParser(description="Run CI-style platform + agent stress campaign.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key", default=os.environ.get("MASTER_API_KEY", ""))
    parser.add_argument("--artifacts-root", default="artifacts/platform_stress")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--max-agents", type=int, default=0, help="0 = all discovered")
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--agent-concurrency", type=int, default=4)
    parser.add_argument("--request-timeout-s", type=int, default=25)
    parser.add_argument("--poll-active-s", type=int, default=15)
    parser.add_argument("--poll-idle-s", type=int, default=60)
    parser.add_argument("--pause-error-rate", type=float, default=0.35)
    parser.add_argument("--pause-seconds", type=int, default=20)
    parser.add_argument("--include-expensive", action="store_true", default=False)
    parser.add_argument("--stress-levels", default="1,2,4")
    parser.add_argument("--stress-rounds", type=int, default=2)
    parser.add_argument("--stress-agent-count", type=int, default=8)
    parser.add_argument("--profile", default="nightly", choices=["nightly", "weekly", "manual"])
    args = parser.parse_args()

    if not args.api_key:
        raise SystemExit("Missing API key. Use --api-key or export MASTER_API_KEY.")

    run_id = args.run_id.strip() or datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg = RunnerConfig(
        base_url=args.base_url,
        api_key=args.api_key,
        artifacts_root=Path(args.artifacts_root),
        run_id=run_id,
        resume=bool(args.resume),
        max_agents=max(0, int(args.max_agents)),
        batch_size=max(1, int(args.batch_size)),
        agent_concurrency=max(1, int(args.agent_concurrency)),
        request_timeout_s=max(5, int(args.request_timeout_s)),
        poll_active_s=max(5, int(args.poll_active_s)),
        poll_idle_s=max(15, int(args.poll_idle_s)),
        pause_error_rate=max(0.0, min(1.0, float(args.pause_error_rate))),
        pause_seconds=max(1, int(args.pause_seconds)),
        include_expensive=bool(args.include_expensive),
        stress_levels=[max(1, int(x)) for x in str(args.stress_levels).split(",") if x.strip()],
        stress_rounds=max(1, int(args.stress_rounds)),
        stress_agent_count=max(1, int(args.stress_agent_count)),
        profile=args.profile,
    )

    run_dir = cfg.artifacts_root / cfg.run_id
    store = ArtifactStore(run_dir)
    checkpoint = store.load_checkpoint()
    client = HttpClient(cfg.base_url, cfg.api_key, cfg.request_timeout_s)

    manifest = {
        "run_id": cfg.run_id,
        "started_at": now_iso(),
        "config": {
            "base_url": cfg.base_url,
            "batch_size": cfg.batch_size,
            "agent_concurrency": cfg.agent_concurrency,
            "request_timeout_s": cfg.request_timeout_s,
            "poll_active_s": cfg.poll_active_s,
            "poll_idle_s": cfg.poll_idle_s,
            "pause_error_rate": cfg.pause_error_rate,
            "pause_seconds": cfg.pause_seconds,
            "include_expensive": cfg.include_expensive,
            "stress_levels": cfg.stress_levels,
            "stress_rounds": cfg.stress_rounds,
            "stress_agent_count": cfg.stress_agent_count,
            "profile": cfg.profile,
            "max_agents": cfg.max_agents,
        },
        "config_hash": {
            "intelligence_config_sha256": sha256_file(ROOT / "intelligence_config.yaml"),
            "agent_config_dir_sha256": sha256_file(ROOT / "agent_config" / "index.json")
            if (ROOT / "agent_config" / "index.json").exists()
            else "",
        },
    }
    store.write_json("run_manifest.json", manifest)
    store.save_checkpoint(checkpoint)

    poll_state = {"active": False}
    stop_poll = threading.Event()

    def poll_loop() -> None:
        while not stop_poll.is_set():
            interval = cfg.poll_active_s if poll_state["active"] else cfg.poll_idle_s
            full = client.request("GET", "/health/full", timeout_s=min(20, cfg.request_timeout_s))
            metrics = client.request("GET", "/metrics", timeout_s=min(20, cfg.request_timeout_s))
            row = {
                "ts": now_iso(),
                "active_mode": poll_state["active"],
                "interval_s": interval,
                "health_full_status": (full.get("json") or {}).get("status"),
                "health_ok": full.get("ok"),
                "metrics_ok": metrics.get("ok"),
                "brain_queue": (((full.get("json") or {}).get("drivers") or {}).get("brain_queue") or {}),
                "local_stream_admission": (((full.get("json") or {}).get("drivers") or {}).get("local_stream_admission") or {}),
                "circuit_breaker": (((full.get("json") or {}).get("drivers") or {}).get("circuit_breaker") or {}),
            }
            store.append_jsonl("polling.jsonl", row)
            stop_poll.wait(interval)

    poll_thread = threading.Thread(target=poll_loop, name="platform-poller", daemon=True)
    poll_thread.start()

    try:
        # Phase 0: preflight
        phase = "phase0_preflight"
        if phase not in checkpoint["completed_phases"]:
            poll_state["active"] = True
            preflight_endpoints = ["/health", "/ready", "/health/full", "/metrics"]
            preflight_rows = []
            for ep in preflight_endpoints:
                res = client.request("GET", ep)
                row = {
                    "phase": phase,
                    "ts": now_iso(),
                    "endpoint": ep,
                    "ok": res["ok"],
                    "status": res["status"],
                    "latency_ms": res["ms"],
                    "status_field": (res.get("json") or {}).get("status"),
                    "error": res["error"],
                }
                preflight_rows.append(row)
                store.append_jsonl("core_checks.jsonl", row)
            checkpoint["completed_phases"].append(phase)
            store.save_checkpoint(checkpoint)
            poll_state["active"] = False

        # Phase 1: inventory + smoke
        phase = "phase1_inventory_smoke"
        agents: list[dict] = []
        if phase not in checkpoint["completed_phases"]:
            poll_state["active"] = True
            inv = client.request("GET", "/v1/agents")
            inv_json = inv.get("json")
            inv_data = None
            if isinstance(inv_json, list):
                inv_data = inv_json
            elif isinstance(inv_json, dict):
                maybe_data = inv_json.get("data")
                if isinstance(maybe_data, list):
                    inv_data = maybe_data
            if not inv["ok"] or inv_data is None:
                raise RuntimeError(f"Failed to fetch agents inventory: status={inv['status']} err={inv['error']}")
            agents = list(inv_data)
            if cfg.max_agents > 0:
                agents = agents[: cfg.max_agents]
            store.write_json("agents_inventory.json", {"count": len(agents), "agents": agents})

            def smoke_one(identifier: str) -> dict:
                body = {
                    "prompt": "Give a short practical recommendation for improving team productivity this week.",
                    "options": {"max_tokens": 220, "temperature": 0.2},
                    "metadata": {},
                }
                r = client.request("POST", f"/v1/agents/{identifier}/run", json_body=body)
                return {
                    "phase": phase,
                    "agent": identifier,
                    "ok": r["ok"],
                    "status": r["status"],
                    "latency_ms": r["ms"],
                    "error": r["error"],
                }

            smoke_rows = []
            ids = [str(a.get("identifier")) for a in agents]
            with concurrent.futures.ThreadPoolExecutor(max_workers=cfg.agent_concurrency) as ex:
                futs = [ex.submit(smoke_one, identifier) for identifier in ids]
                for fut in concurrent.futures.as_completed(futs):
                    row = fut.result()
                    smoke_rows.append(row)
                    store.append_jsonl("smoke.jsonl", row)

            checkpoint["completed_phases"].append(phase)
            store.save_checkpoint(checkpoint)
            poll_state["active"] = False
        else:
            inv_path = run_dir / "agents_inventory.json"
            payload = safe_json_loads(inv_path.read_text()) if inv_path.exists() else {}
            agents = list((payload or {}).get("agents") or [])

        # Prepare matrix cases for phase2
        all_cases = build_case_matrix(agents)
        store.write_json("cases_manifest.json", {"count": len(all_cases), "cases": all_cases})

        # Phase 2: full matrix
        phase = "phase2_agent_matrix"
        if phase not in checkpoint["completed_phases"]:
            poll_state["active"] = True
            completed = set(checkpoint.get("completed_case_ids") or [])
            pending = [c for c in all_cases if c["case_id"] not in completed]

            def exec_case(case: dict) -> dict:
                body = {"prompt": case["prompt"], "options": case.get("options") or {}, "metadata": {}}
                res = client.request(
                    "POST",
                    f"/v1/agents/{case['agent']}/run",
                    json_body=body,
                    timeout_s=cfg.request_timeout_s,
                )
                expected_success = bool(case["expect_success"])
                observed_success = bool(res["ok"])
                passed = observed_success if expected_success else (not observed_success)
                out = {
                    "ts": now_iso(),
                    "case_id": case["case_id"],
                    "agent": case["agent"],
                    "case_type": case["case_type"],
                    "expected_success": expected_success,
                    "observed_success": observed_success,
                    "passed": passed,
                    "status": res["status"],
                    "latency_ms": res["ms"],
                    "error": res["error"],
                    "response_excerpt": (res.get("raw") or "")[:400],
                }
                return out

            for i in range(0, len(pending), cfg.batch_size):
                batch = pending[i : i + cfg.batch_size]
                rows: list[dict] = []
                with concurrent.futures.ThreadPoolExecutor(max_workers=cfg.agent_concurrency) as ex:
                    futs = [ex.submit(exec_case, c) for c in batch]
                    for fut in concurrent.futures.as_completed(futs):
                        row = fut.result()
                        rows.append(row)
                        store.append_jsonl("cases.jsonl", row)
                        checkpoint["completed_case_ids"].append(row["case_id"])
                        store.save_checkpoint(checkpoint)

                expected_success_rows = [r for r in rows if r["expected_success"]]
                unexpected_fail = [r for r in expected_success_rows if not r["passed"]]
                error_rate = (len(unexpected_fail) / len(expected_success_rows)) if expected_success_rows else 0.0
                store.append_jsonl(
                    "events.jsonl",
                    {
                        "ts": now_iso(),
                        "phase": phase,
                        "batch_start": i,
                        "batch_size": len(batch),
                        "unexpected_error_rate": round(error_rate, 4),
                        "pause_triggered": error_rate > cfg.pause_error_rate,
                    },
                )
                if error_rate > cfg.pause_error_rate:
                    time.sleep(cfg.pause_seconds)

            checkpoint["completed_phases"].append(phase)
            store.save_checkpoint(checkpoint)
            poll_state["active"] = False

        # Phase 3: core platform endpoint checks
        phase = "phase3_core_checks"
        if phase not in checkpoint["completed_phases"]:
            poll_state["active"] = True
            core_cases = [
                {"name": "chat_sync", "method": "POST", "path": "/v1/chat", "json": {"agent_type": "personal_assistant", "prompt": "Say hello briefly.", "options": {}}, "expect_success": True},
                {"name": "chat_stream", "method": "POST", "path": "/v1/chat/stream", "json": {"agent_type": "personal_assistant", "prompt": "Stream one short answer.", "options": {}}, "expect_success": True, "stream_hint": True},
                {"name": "int_sentiment", "method": "POST", "path": "/v1/intelligence/sentiment", "json": {"text": "I love this product and feel great using it."}, "expect_success": True},
                {"name": "int_language", "method": "POST", "path": "/v1/intelligence/language", "json": {"text": "Namaste doston kaise ho"}, "expect_success": True},
                {"name": "int_languages", "method": "GET", "path": "/v1/intelligence/languages", "expect_success": True},
                {"name": "rag_query", "method": "POST", "path": "/v1/rag/query", "query": {"query": "test", "n_results": 1}, "expect_success": True},
                {"name": "feedback_stats", "method": "GET", "path": "/v1/feedback/stats", "expect_success": True},
                {"name": "billing_summary", "method": "GET", "path": "/v1/billing/summary", "expect_success": True},
                {"name": "voip_personas", "method": "GET", "path": "/v1/voip/personas", "expect_success": True},
                {"name": "voip_compliance", "method": "GET", "path": "/v1/voip/compliance", "expect_success": True},
                {"name": "int_url_analyze", "method": "POST", "path": "/v1/intelligence/url-analyze", "json": {"url": "https://example.com", "max_pages": 1}, "expect_success": True},
            ]
            if cfg.include_expensive:
                core_cases.append(
                    {"name": "images_generate", "method": "POST", "path": "/v1/images/generate", "json": {"prompt": "minimal logo icon, simple monochrome"}, "expect_success": True}
                )
            else:
                store.append_jsonl(
                    "core_checks.jsonl",
                    {
                        "phase": phase,
                        "name": "images_generate",
                        "skipped": True,
                        "reason": "include_expensive=false",
                        "ts": now_iso(),
                    },
                )

            for c in core_cases:
                res = client.request(
                    c["method"],
                    c["path"],
                    json_body=c.get("json"),
                    query=c.get("query"),
                    timeout_s=cfg.request_timeout_s,
                    stream_hint=bool(c.get("stream_hint")),
                )
                passed = bool(res["ok"]) if c["expect_success"] else (not bool(res["ok"]))
                row = {
                    "phase": phase,
                    "ts": now_iso(),
                    "name": c["name"],
                    "method": c["method"],
                    "path": c["path"],
                    "expect_success": c["expect_success"],
                    "observed_success": res["ok"],
                    "passed": passed,
                    "status": res["status"],
                    "latency_ms": res["ms"],
                    "error": res["error"],
                    "response_excerpt": (res.get("raw") or "")[:400],
                }
                store.append_jsonl("core_checks.jsonl", row)

            checkpoint["completed_phases"].append(phase)
            store.save_checkpoint(checkpoint)
            poll_state["active"] = False

        # Phase 4: stress slices
        phase = "phase4_stress_slices"
        if phase not in checkpoint["completed_phases"]:
            poll_state["active"] = True
            ids = [str(a.get("identifier")) for a in agents]
            heavy_keywords = ("analyzer", "strategist", "generator", "planner", "advisor", "optimizer", "research")
            heavy = [x for x in ids if any(k in x for k in heavy_keywords)]
            if len(heavy) < cfg.stress_agent_count:
                for x in ids:
                    if x not in heavy:
                        heavy.append(x)
                    if len(heavy) >= cfg.stress_agent_count:
                        break
            heavy = heavy[: cfg.stress_agent_count]

            def stress_one(agent: str) -> dict:
                body = {
                    "prompt": "Provide a concise, actionable plan for a weekly growth sprint with 3 priorities.",
                    "options": {"max_tokens": 300, "temperature": 0.2},
                    "metadata": {},
                }
                r = client.request("POST", f"/v1/agents/{agent}/run", json_body=body, timeout_s=cfg.request_timeout_s)
                return {"agent": agent, "ok": bool(r["ok"]), "status": r["status"], "ms": r["ms"], "error": r["error"]}

            for level in cfg.stress_levels:
                tasks = [random.choice(heavy) for _ in range(max(1, len(heavy) * cfg.stress_rounds))]
                rows = []
                with concurrent.futures.ThreadPoolExecutor(max_workers=level) as ex:
                    futs = [ex.submit(stress_one, a) for a in tasks]
                    for fut in concurrent.futures.as_completed(futs):
                        rows.append(fut.result())
                ok_rows = [r for r in rows if r["ok"]]
                lat = [float(r["ms"]) for r in ok_rows]
                summary = {
                    "phase": phase,
                    "ts": now_iso(),
                    "concurrency": level,
                    "requests": len(rows),
                    "success": len(ok_rows),
                    "failure": len(rows) - len(ok_rows),
                    "success_rate_pct": round((len(ok_rows) / len(rows)) * 100, 2) if rows else 0.0,
                    "p50_ms": percentile(lat, 50),
                    "p95_ms": percentile(lat, 95),
                    "avg_ms": round(statistics.mean(lat), 2) if lat else None,
                }
                store.append_jsonl("stress_slices.jsonl", summary)
                if rows and (1 - (len(ok_rows) / len(rows))) > cfg.pause_error_rate:
                    time.sleep(cfg.pause_seconds)

            checkpoint["completed_phases"].append(phase)
            store.save_checkpoint(checkpoint)
            poll_state["active"] = False

        # Phase 5: summary artifacts
        phase = "phase5_summary"
        if phase not in checkpoint["completed_phases"]:
            def load_jsonl(rel: str) -> list[dict]:
                p = run_dir / rel
                if not p.exists():
                    return []
                out = []
                for ln in p.read_text(encoding="utf-8").splitlines():
                    row = safe_json_loads(ln)
                    if isinstance(row, dict):
                        out.append(row)
                return out

            case_rows = load_jsonl("cases.jsonl")
            core_rows = [r for r in load_jsonl("core_checks.jsonl") if not r.get("skipped")]
            poll_rows = load_jsonl("polling.jsonl")
            stress_rows = load_jsonl("stress_slices.jsonl")

            discovered_agents = {a.get("identifier") for a in agents}
            completed_agents = {r.get("agent") for r in case_rows if isinstance(r.get("agent"), str)}
            failed_cases = [r for r in case_rows if not bool(r.get("passed"))]
            failed_core = [r for r in core_rows if not bool(r.get("passed"))]
            failed_case_ids = [r.get("case_id") for r in failed_cases if r.get("case_id")]

            success_case_lat = [float(r["latency_ms"]) for r in case_rows if r.get("observed_success") and isinstance(r.get("latency_ms"), (int, float))]
            timeout_count = len([r for r in case_rows if "timed out" in str(r.get("error", "")).lower()])
            degraded_intervals = len([r for r in poll_rows if str(r.get("health_full_status")) not in ("ok", "None", "")])

            summary = {
                "run_id": cfg.run_id,
                "completed_at": now_iso(),
                "agents": {
                    "discovered": len(discovered_agents),
                    "covered": len(completed_agents),
                    "coverage_pct": round((len(completed_agents) / len(discovered_agents)) * 100, 2) if discovered_agents else 0.0,
                },
                "cases": {
                    "total": len(case_rows),
                    "passed": len(case_rows) - len(failed_cases),
                    "failed": len(failed_cases),
                    "pass_pct": round(((len(case_rows) - len(failed_cases)) / len(case_rows)) * 100, 2) if case_rows else 0.0,
                    "timeout_rate_pct": round((timeout_count / len(case_rows)) * 100, 2) if case_rows else 0.0,
                },
                "core_endpoints": {
                    "total": len(core_rows),
                    "passed": len(core_rows) - len(failed_core),
                    "failed": len(failed_core),
                    "pass_pct": round(((len(core_rows) - len(failed_core)) / len(core_rows)) * 100, 2) if core_rows else 0.0,
                },
                "latency_ms": {
                    "p50": percentile(success_case_lat, 50),
                    "p95": percentile(success_case_lat, 95),
                    "avg": round(statistics.mean(success_case_lat), 2) if success_case_lat else None,
                },
                "polling": {
                    "snapshots": len(poll_rows),
                    "degraded_intervals": degraded_intervals,
                },
                "stress_slices": stress_rows,
                "rerun_targets": {
                    "case_ids": failed_case_ids[:500],
                    "core_checks": [r.get("name") for r in failed_core],
                },
            }

            failures = {
                "cases": failed_cases,
                "core": failed_core,
            }
            store.write_json("summary.json", summary)
            store.write_json("failures.json", failures)

            md_lines = [
                f"# Platform Stress Summary ({cfg.run_id})",
                "",
                f"- Profile: `{cfg.profile}`",
                f"- Base URL: `{cfg.base_url}`",
                f"- Agent coverage: **{summary['agents']['covered']}/{summary['agents']['discovered']} ({summary['agents']['coverage_pct']}%)**",
                f"- Case pass: **{summary['cases']['passed']}/{summary['cases']['total']} ({summary['cases']['pass_pct']}%)**",
                f"- Core endpoint pass: **{summary['core_endpoints']['passed']}/{summary['core_endpoints']['total']} ({summary['core_endpoints']['pass_pct']}%)**",
                f"- Latency p50/p95 (success cases): **{summary['latency_ms']['p50']}ms / {summary['latency_ms']['p95']}ms**",
                f"- Timeout rate: **{summary['cases']['timeout_rate_pct']}%**",
                f"- Polling degraded intervals: **{summary['polling']['degraded_intervals']}**",
                "",
                "## Rerun Targets",
                f"- Failed case IDs: {len(summary['rerun_targets']['case_ids'])}",
                f"- Failed core checks: {len(summary['rerun_targets']['core_checks'])}",
            ]
            (run_dir / "summary.md").write_text("\n".join(md_lines))

            checkpoint["completed_phases"].append(phase)
            store.save_checkpoint(checkpoint)

    finally:
        stop_poll.set()
        poll_thread.join(timeout=5)

    print(json.dumps({"run_id": cfg.run_id, "artifacts_dir": str(run_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
