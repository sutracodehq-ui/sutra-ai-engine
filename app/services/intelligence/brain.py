"""
Brain — Unified execution pipeline for the SutraCode AI Engine.

Software Factory: Everything is config-driven via intelligence_config.yaml.
Local-first → quality gate → cloud escalation.

Core APIs:
    route()              — O(1) driver + model selection
    execute()            — local-first → quality gate → cloud
    execute_speculative  — draft-verify pattern
    execute_swarm        — multi-agent decomposition
    chain()              — agent review chains
    select_prompt()      — OPRO champion/candidate selection
    filter_response()    — raw → AgentResult normalization
    queue                — semaphore-based concurrency
"""

import asyncio
import hashlib
import json
import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncGenerator, Callable, Optional, Tuple

import yaml
from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.requests import Request

from app.config import get_settings
from app.models.agent_optimization import AgentOptimization
from app.schemas.agent_result import AgentResult
from app.services.drivers.base import LlmResponse

logger = logging.getLogger(__name__)

# ─── Config (cached 60s) ──────────────────────────────────────

_config_cache: dict | None = None
_config_ts: float = 0
_indic_scripts: frozenset | None = None
_hinglish_words: frozenset | None = None
_complex_signals: frozenset | None = None
_simple_signals: frozenset | None = None


def _load_config() -> dict:
    global _config_cache, _config_ts, _indic_scripts, _hinglish_words, _complex_signals, _simple_signals
    now = time.monotonic()
    if _config_cache is not None and (now - _config_ts) < 60.0:
        return _config_cache
    path = Path("intelligence_config.yaml")
    _config_cache = yaml.safe_load(open(path)) if path.exists() else {}
    _config_ts = now
    _indic_scripts = _hinglish_words = _complex_signals = _simple_signals = None
    return _config_cache


def _cfg(section: str, key: str = None, default=None):
    data = _load_config().get(section, {})
    return data.get(key, default) if key else (data or default)


def _ensure_sets():
    global _indic_scripts, _hinglish_words, _complex_signals, _simple_signals
    if _indic_scripts is not None:
        return
    sr = _cfg("smart_router", default={})
    _indic_scripts = frozenset(sr.get("indic_scripts", []))
    _hinglish_words = frozenset(sr.get("hinglish_words", []))
    _complex_signals = frozenset(sr.get("complex_signals", []))
    _simple_signals = frozenset(sr.get("simple_signals", []))


from app.services.intelligence.multilingual import detect_language


# ─── Response Filter ──────────────────────────────────────────

class _ResponseFilter:
    """Normalizes raw LLM text → AgentResult."""
    STANDARD_FIELDS = {"suggestions"}

    def filter(self, raw: str, agent_config: dict | None = None) -> AgentResult:
        if not raw or not raw.strip():
            return AgentResult(data={"content": ""}, suggestions=[], raw=raw or "", parsed=False)
        parsed, ok = self._try_json(raw)
        if not ok or not isinstance(parsed, dict):
            return AgentResult(data={"content": raw}, suggestions=[], raw=raw, parsed=False)
        suggestions = [str(s) for s in parsed.pop("suggestions", []) if s] if isinstance(parsed.get("suggestions"), list) else []
        if agent_config:
            fields = agent_config.get("response_schema", {})
            fields = fields.get("fields", []) if isinstance(fields, dict) else fields if isinstance(fields, list) else []
            missing = [f for f in fields if f not in self.STANDARD_FIELDS and f not in parsed]
            if missing:
                logger.warning(f"Brain.filter: missing fields: {missing}")
        return AgentResult(data=parsed, suggestions=suggestions, raw=raw, parsed=True)

    def _try_json(self, raw: str) -> tuple[Any, bool]:
        import re
        for text in [raw.strip(), re.sub(r"```(?:json)?\s*\n?(.*?)\n?\s*```", r"\1", raw.strip(), flags=re.DOTALL).strip()]:
            try:
                return json.loads(text), True
            except json.JSONDecodeError:
                pass
        # Extract first JSON object
        start = raw.find("{")
        if start == -1:
            return None, False
        depth, in_str, esc = 0, False, False
        for i in range(start, len(raw)):
            ch = raw[i]
            if esc: esc = False; continue
            if ch == "\\": esc = True; continue
            if ch == '"': in_str = not in_str; continue
            if in_str: continue
            if ch == "{": depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try: return json.loads(raw[start:i+1]), True
                    except json.JSONDecodeError: return None, False
        return None, False


# ─── LLM Queue ────────────────────────────────────────────────

def _sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


class _LlmQueue:
    """Async semaphore queue with SSE status events."""
    def __init__(self, max_parallel: int | None = None):
        self._max = max_parallel or get_settings().llm_max_parallel
        self._sem = asyncio.Semaphore(self._max)
        self._active = self._waiting = 0

    @property
    def stats(self) -> dict:
        return {"max_parallel": self._max, "active": self._active, "waiting": self._waiting}

    async def stream(self, gen_fn: Callable[[], AsyncGenerator[str, None]], request: Request | None = None) -> AsyncGenerator[str, None]:
        self._waiting += 1
        yield _sse({"type": "status", "stage": "thinking"})
        while not self._sem._value > 0:
            if request and await request.is_disconnected():
                self._waiting -= 1; return
            try:
                await asyncio.wait_for(self._sem.acquire(), timeout=2.0); break
            except asyncio.TimeoutError: continue
            except asyncio.CancelledError: self._waiting -= 1; return
        else:
            await self._sem.acquire()
        self._waiting -= 1; self._active += 1
        cancelled = False
        try:
            yield _sse({"type": "status", "stage": "calculating"})
            gen = gen_fn(); count = 0
            try:
                async for chunk in gen:
                    count += 1
                    if request and count % 5 == 0 and await request.is_disconnected():
                        cancelled = True; break
                    yield chunk
            finally:
                if cancelled: await gen.aclose()
        except asyncio.CancelledError: pass
        except Exception as e:
            logger.error(f"Brain.queue: {e}")
            yield _sse({"type": "error", "message": str(e)})
        finally:
            self._active -= 1; self._sem.release()


# ─── CoT Injection ────────────────────────────────────────────

THINKING_INJECTION = """
Before responding, think through: 1) What is requested? 2) What context exists?
3) Best strategy for quality output? 4) Is the output complete and accurate?
Do NOT include your thinking — only the final result.
"""


# ─── Brain ────────────────────────────────────────────────────

class Brain:
    """Config-driven execution pipeline. All thresholds from YAML."""

    def __init__(self):
        self._filter = _ResponseFilter()
        self._queue = _LlmQueue()

    @property
    def queue(self) -> _LlmQueue:
        return self._queue

    # ── Routing ───────────────────────────────────────────────

    async def route(self, prompt: str, agent_type: str, circuit_breaker=None) -> dict:
        """O(1) routing: language → scout → driver → model."""
        s = get_settings()
        if not s.ai_smart_router_enabled:
            return {"driver": s.ai_driver, "model": None, "complexity": "moderate",
                    "language": "english", "reason": "router off", "chain": None,
                    "intent": "general", "needs_web": False}

        language = detect_language(prompt)
        scout = await self._assess_complexity(prompt, agent_type)
        complexity, intent, needs_web = scout["complexity"], scout.get("intent", "general"), scout.get("needs_web", False)
        chain = self._get_chain(complexity, language)
        from app.services.intelligence.guardian import get_guardian
        route_hint = await get_guardian().get_route_hint(agent_type)
        chain = self._order_chain_for_route_hint(chain, route_hint)
        driver = self._pick_driver(complexity, language, circuit_breaker, chain_override=chain)
        scout_score = scout.get("scout_score")
        model = self._pick_model(driver, complexity, scout_score=scout_score)

        parts = []
        if language != "english": parts.append(f"{language}")
        parts.append(f"{complexity}→{driver}")
        if model: parts.append(f"({model})")
        if needs_web: parts.append("[web]")
        reason = " ".join(parts)
        logger.info(f"Brain.route: {reason} intent={intent}")

        return {"driver": driver, "model": model, "complexity": complexity,
                "language": language, "reason": reason, "chain": chain,
                "intent": intent, "needs_web": needs_web}

    def _get_chain(self, complexity: str, language: str) -> list[str]:
        sr = _cfg("smart_router", default={})
        lang_key = "english" if language == "english" else "indic"
        return sr.get("driver_chains", {}).get(lang_key, {}).get(complexity, ["ollama", "groq"])

    async def _assess_complexity(self, prompt: str, agent_type: str) -> dict:
        """Scout LLM (primary) → heuristic fallback."""
        scout_cfg = _cfg("smart_router", default={}).get("scout", {})
        if scout_cfg.get("enabled", False):
            try:
                result = await self._scout_score(prompt, scout_cfg)
                if result:
                    logger.info(f"Brain: Scout → {result['complexity']}, intent={result['intent']}, web={result['needs_web']}")
                    return result
            except Exception as e:
                logger.debug(f"Brain: Scout failed ({e}), heuristic")
        return {
            "complexity": self._heuristic_complexity(prompt, agent_type),
            "intent": "general",
            "needs_web": False,
            "scout_score": None,
        }

    @staticmethod
    def _order_chain_for_route_hint(chain: list[str], route_hint: str) -> list[str]:
        """Reorder driver chain using Redis quality routing hints (never blocks routing)."""
        locals_first = frozenset({"ollama", "bitnet"})
        if not chain:
            return chain
        if route_hint == "fast_local":
            return [d for d in chain if d in locals_first] + [d for d in chain if d not in locals_first]
        if route_hint == "direct_cloud":
            return [d for d in chain if d not in locals_first] + [d for d in chain if d in locals_first]
        return list(chain)

    async def _scout_score(self, prompt: str, cfg: dict) -> dict | None:
        """Scout LLM; any failure returns None so heuristic routing always applies."""
        from app.services.llm_service import get_llm_service
        tiers = cfg.get("complexity_tiers", {"simple": [1, 3], "moderate": [4, 6], "complex": [7, 10]})
        try:
            resp = await asyncio.wait_for(
                get_llm_service().complete(
                    prompt=prompt[: cfg.get("max_prompt_chars", 300)],
                    system_prompt=cfg.get("system_prompt", "Classify task complexity 1-10 as JSON."),
                    driver=cfg.get("driver", "ollama"),
                    model=cfg.get("model", "qwen3:1.7b"),
                    max_tokens=100,
                    temperature=0.0,
                ),
                timeout=cfg.get("timeout_ms", 2000) / 1000.0,
            )
        except (asyncio.TimeoutError, Exception) as e:
            logger.debug(f"Brain: Scout transport failed ({e}), heuristic routing")
            return None
        raw = (resp.content or "").strip()
        if not raw:
            logger.debug("Brain: Scout empty body, heuristic routing")
            return None
        text = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            logger.debug("Brain: Scout invalid JSON, heuristic routing")
            return None
        if not isinstance(data, dict):
            return None
        try:
            score = int(data.get("complexity", 5))
        except (TypeError, ValueError):
            score = 5
        score = max(1, min(10, score))
        tier = next((t for t, (lo, hi) in tiers.items() if lo <= score <= hi), "moderate")
        intent = data.get("intent", "general")
        if not isinstance(intent, str):
            intent = "general"
        needs_web = bool(data.get("needs_web", False))
        return {
            "complexity": tier,
            "intent": intent,
            "needs_web": needs_web,
            "scout_score": score,
        }

    def _heuristic_complexity(self, prompt: str, agent_type: str) -> str:
        _ensure_sets()
        sr = _cfg("smart_router", default={})
        wc = len(prompt.split(None, 100))
        buckets = sr.get("length_buckets", {"short": 10, "long": 80})
        bucket = "short" if wc < buckets.get("short", 10) else ("long" if wc > buckets.get("long", 80) else "medium")
        n = sr.get("signal_words", 3)
        words = prompt.lower().split(None, n + 1)[:n]
        signal = "none"
        for w in words:
            if w in _complex_signals: signal = "complex"; break
            if w in _simple_signals: signal = "simple"
        agent_tier = "moderate"
        try:
            from app.services.agents.hub import get_agent_hub
            agent_tier = get_agent_hub().get(agent_type)._config.get("complexity", "moderate")
        except Exception: pass
        return sr.get("decision_table", {}).get(f"{bucket}_{signal}_{agent_tier}", sr.get("default_complexity", "moderate"))

    def _pick_driver(self, complexity: str, language: str, cb=None, chain_override: list[str] | None = None) -> str:
        chain = chain_override if chain_override is not None else self._get_chain(complexity, language)
        s = get_settings()
        key_map = {
            "openai": bool(s.openai_api_key),
            "anthropic": bool(s.anthropic_api_key),
            "gemini": bool(s.gemini_api_key),
            "groq": bool(s.groq_api_key),
            "sarvam": bool(s.sarvam_api_key),
            "nvidia": bool(s.nvidia_api_key),
            "bitnet": True,
            "ollama": True,
        }
        for d in chain:
            if key_map.get(d, False) and (not cb or cb.is_available(d)):
                return d
        for d in chain:
            if key_map.get(d, False):
                return d
        return s.ai_driver

    def _pick_model(
        self, driver: str, complexity: str, scout_score: int | None = None
    ) -> Optional[str]:
        """Pick model tier; nudge toward heavier tier when Scout numeric score disagrees with bucket."""
        tiers = _cfg("smart_router", default={}).get("model_tiers", {})
        eff = complexity
        if scout_score is not None:
            if scout_score >= 8 and complexity == "simple":
                eff = "moderate"
            elif scout_score >= 8 and complexity == "moderate":
                eff = "complex"
            elif scout_score <= 2 and complexity == "complex":
                eff = "moderate"
            elif scout_score <= 2 and complexity == "moderate":
                eff = "simple"
        model = tiers.get(driver, {}).get(eff) or tiers.get(driver, {}).get(complexity)
        if not model:
            s = get_settings()
            model = {"ollama": s.ollama_model, "sarvam": s.sarvam_model, "nvidia": s.nvidia_model}.get(driver)
        return model

    async def select_model(self, prompt: str, agent_type: str, driver: str) -> Optional[str]:
        sc = await self._assess_complexity(prompt, agent_type)
        return self._pick_model(driver, sc["complexity"], scout_score=sc.get("scout_score"))

    # ── Execution ─────────────────────────────────────────────

    async def execute(self, prompt: str, system_prompt: str, agent_type: str,
                      expected_fields: list[str] | None = None, **options) -> LlmResponse:
        """Local-first → quality gate → cloud escalation."""
        from app.services.intelligence.guardian import get_guardian

        scout = await self._assess_complexity(prompt, agent_type)
        complexity = scout["complexity"]
        needs_web = scout.get("needs_web", False)
        scout_score = scout.get("scout_score")
        system_prompt = self._inject_thinking(system_prompt, complexity, agent_type)

        # Web augmentation
        if needs_web:
            ctx = await self._augment_with_web(prompt)
            if ctx: system_prompt = f"{system_prompt}\n\n--- LIVE WEB CONTEXT ---\n{ctx}"

        # Peer teaching
        alliance_ctx = await self._get_alliance_traces(prompt, agent_type)
        if alliance_ctx: system_prompt = f"{system_prompt}\n\n--- PEER KNOWLEDGE ---\n{alliance_ctx}"

        # Consensus for complex
        consensus_cfg = _cfg("smart_router", default={}).get("consensus", {})
        if consensus_cfg.get("enabled", False) and complexity == consensus_cfg.get("min_complexity_tier", "complex"):
            try:
                resp = await self._execute_consensus(prompt, system_prompt, agent_type, consensus_cfg, **options)
                if resp and resp.content: return resp
            except Exception as e:
                logger.warning(f"Brain: consensus failed ({e}), single model")

        if not get_settings().ai_hybrid_routing:
            out = await self._call_driver(prompt, system_prompt, **options)
            return await self._finalize_nonempty_response(prompt, system_prompt, out, **options)

        guardian = get_guardian()
        route_hint = await guardian.get_route_hint(agent_type)

        if route_hint == "direct_cloud":
            resp = await self._call_cloud(
                prompt, system_prompt, agent_type=agent_type, scout_score=scout_score, **options
            )
            await self._auto_train(agent_type, prompt, resp)
            return await self._finalize_nonempty_response(prompt, system_prompt, resp, **options)

        start = time.monotonic()
        local_resp = await self._call_local(prompt, system_prompt, **options)
        latency = round((time.monotonic() - start) * 1000)

        if route_hint == "fast_local":
            await guardian.record_quality(agent_type, 8.0)
            return await self._finalize_nonempty_response(prompt, system_prompt, local_resp, **options)

        score_result = guardian.score_response(local_resp, expected_fields)
        score = score_result["total"]
        if score_result["passed"]:
            await guardian.record_quality(agent_type, score)
            return await self._finalize_nonempty_response(prompt, system_prompt, local_resp, **options)

        logger.info(f"Brain: ESCALATING {agent_type} (score={score})")
        cloud_resp = await self._call_cloud(
            prompt, system_prompt, agent_type=agent_type, scout_score=scout_score, **options
        )
        cloud_score = guardian.score_response(cloud_resp, expected_fields)["total"]
        await guardian.record_quality(agent_type, score)
        if cloud_score >= score_result["threshold"]:
            await self._auto_train(agent_type, prompt, cloud_resp)
            return await self._finalize_nonempty_response(prompt, system_prompt, cloud_resp, **options)
        chosen = cloud_resp if cloud_score > score else local_resp
        return await self._finalize_nonempty_response(prompt, system_prompt, chosen, **options)

    async def _finalize_nonempty_response(
        self, prompt: str, system_prompt: str, resp: LlmResponse, **opts
    ) -> LlmResponse:
        """Registry-wide chain + offline emergency — never return an empty body to callers."""
        if resp and (resp.content or "").strip():
            return resp
        logger.warning("Brain: empty model output; running global driver chain + emergency fallback")
        try:
            from app.services.llm_service import get_llm_service
            call_opts = {k: v for k, v in opts.items() if k != "agent_type"}
            return await get_llm_service().complete(
                prompt=prompt, system_prompt=system_prompt, driver=None, **call_opts
            )
        except Exception as e:
            logger.warning(f"Brain: registry finalize failed ({e}), offline synthesizer")
        from app.services.intelligence.driver import EmergencyFallbackDriver
        return await EmergencyFallbackDriver().complete(system_prompt, prompt)

    async def _call_local(self, prompt: str, system_prompt: str, **opts) -> LlmResponse:
        from app.services.intelligence.guardian import get_guardian
        from app.services.llm_service import get_llm_service
        if not get_guardian().circuit_breaker.is_available("ollama"):
            return LlmResponse(content="", total_tokens=0, driver="ollama", model="n/a", metadata={"error": "circuit_open"})
        try:
            return await get_llm_service().complete(prompt=prompt, system_prompt=system_prompt, driver="ollama", **opts)
        except Exception as e:
            logger.warning(f"Brain: local failed: {e}")
            return LlmResponse(content="", total_tokens=0, driver="ollama", model="n/a", metadata={"error": str(e)})

    async def _call_cloud(self, prompt: str, system_prompt: str, **opts) -> LlmResponse:
        from app.services.llm_service import get_llm_service
        svc = get_llm_service()
        opts.pop("agent_type", "unknown")
        scout_score = opts.pop("scout_score", None)
        chain = self._get_chain("complex", "english")
        cloud_drivers = [d for d in chain if d not in ("ollama", "bitnet")]
        for driver in cloud_drivers:
            try:
                model = self._pick_model(driver, "complex", scout_score=scout_score)
                resp = await svc.complete(
                    prompt=prompt, system_prompt=system_prompt, driver=driver, model=model, **opts
                )
                if not (resp.content or "").strip():
                    logger.warning(f"Brain: cloud {driver} returned empty body, trying next")
                    continue
                resp.metadata = resp.metadata or {}
                resp.metadata["cloud_driver"] = driver
                return resp
            except Exception as e:
                logger.warning(f"Brain: cloud {driver} failed: {e}")
        logger.warning("Brain: all cloud drivers empty or failed; full registry chain")
        try:
            return await svc.complete(prompt=prompt, system_prompt=system_prompt, driver=None, **opts)
        except Exception as e:
            logger.warning(f"Brain: registry chain after cloud failed: {e}")
        from app.services.intelligence.driver import EmergencyFallbackDriver
        return await EmergencyFallbackDriver().complete(system_prompt, prompt)

    async def _call_driver(self, prompt: str, system_prompt: str, **opts) -> LlmResponse:
        from app.services.llm_service import get_llm_service
        return await get_llm_service().complete(prompt=prompt, system_prompt=system_prompt, **opts)

    async def _auto_train(self, agent_type: str, prompt: str, resp: LlmResponse):
        if not resp.content: return
        try:
            from app.services.intelligence.memory import get_memory
            await get_memory().remember(agent_type, prompt, resp.content, quality_score=1.0)
        except Exception as e:
            logger.warning(f"Brain: auto-train failed: {e}")

    # ── Consensus (multi-model) ───────────────────────────────

    async def _execute_consensus(self, prompt: str, system_prompt: str, agent_type: str,
                                  cfg: dict, **options) -> LlmResponse:
        from app.services.llm_service import get_llm_service
        svc = get_llm_service()
        drivers = cfg.get("parallel_drivers", ["groq", "gemini"])

        async def _call_one(drv):
            try:
                model = self._pick_model(drv, "complex")
                return drv, await svc.complete(prompt=prompt, system_prompt=system_prompt, driver=drv, model=model, **options)
            except Exception as e:
                logger.warning(f"Brain.consensus: {drv} failed: {e}")
                return drv, LlmResponse(content="", total_tokens=0, driver=drv, model="error")

        try:
            results = await asyncio.wait_for(asyncio.gather(*[_call_one(d) for d in drivers]), timeout=cfg.get("timeout_s", 30))
        except asyncio.TimeoutError:
            return None

        valid = [(d, r) for d, r in results if r.content]
        if not valid: return None
        if len(valid) == 1:
            valid[0][1].metadata = {"consensus": "single_valid"}; return valid[0][1]

        # Judge
        judge_driver, judge_model = cfg.get("judge_driver", "groq"), cfg.get("judge_model", "llama-3.1-8b-instant")
        comparison = f"PROMPT: {prompt[:500]}\n\n"
        for i, (drv, resp) in enumerate(valid[:2]):
            comparison += f"RESPONSE {'AB'[i]} ({drv}):\n{resp.content[:2000]}\n\n"
        try:
            judge_resp = await get_llm_service().complete(prompt=comparison, system_prompt=cfg.get("judge_prompt", "Compare two responses."),
                                                          driver=judge_driver, model=judge_model, max_tokens=200, temperature=0.0)
            data = json.loads(judge_resp.content.strip().removeprefix("```json").removesuffix("```").strip())
            idx = 0 if data.get("winner", "A") == "A" else min(1, len(valid) - 1)
            winner = valid[idx][1]
            winner.metadata = {"consensus": {"winner": valid[idx][0], "confidence": data.get("confidence", 0.5)}}
            await self._record_model_win(valid[idx][0], agent_type)
            return winner
        except Exception:
            return valid[0][1]

    async def _record_model_win(self, driver: str, agent_type: str):
        try:
            from app.services.intelligence.memory import get_memory
            await get_memory().remember("model_wins", f"{driver}:{agent_type}",
                                        json.dumps({"driver": driver, "agent": agent_type, "ts": time.time()}), quality_score=1.0)
        except Exception: pass

    # ── Knowledge Augmentation ────────────────────────────────

    async def _augment_with_web(self, prompt: str) -> str | None:
        ka = _cfg("smart_router", default={}).get("knowledge_augment", {})
        if not ka.get("enabled", False): return None
        try:
            from app.services.intelligence.memory import get_memory
            result = await asyncio.wait_for(get_memory().web_search(prompt, max_results=3), timeout=ka.get("timeout_s", 10))
            if not result.get("results"): return None
            parts = [f"Summary: {result['answer'][:500]}"] if result.get("answer") else []
            parts.extend(f"[{r.get('title', '')}]: {r.get('snippet', '')}" for r in result["results"][:3])
            return "\n".join(parts)[:ka.get("max_context_chars", 2000)]
        except Exception as e:
            logger.warning(f"Brain.web_augment: {e}"); return None

    async def _get_alliance_traces(self, prompt: str, agent_type: str) -> str | None:
        try:
            from app.services.intelligence.memory import get_memory
            alliances = _cfg("agent_teaching", default={}).get("alliances", {})
            allies = []
            for _, cfg in alliances.items():
                if agent_type in cfg.get("members", []):
                    allies = [m for m in cfg["members"] if m != agent_type]; break
            if not allies: return None
            min_q = _cfg("agent_teaching", default={}).get("min_quality_to_teach", 4.0)
            traces = []
            for ally in allies[:3]:
                for r in (await get_memory().recall(ally, prompt, n=1) or []):
                    if r.get("meta", {}).get("quality", 0) >= min_q:
                        traces.append(f"[{ally}]: {r['content'][:500]}")
            return "\n".join(traces[:2]) if traces else None
        except Exception: return None

    async def _record_gold_trace(self, agent_type: str, prompt: str, response: str, score: float):
        if score < 7.0: return
        try:
            from app.services.intelligence.memory import get_memory
            await get_memory().remember(agent_type, prompt, response, quality_score=score)
        except Exception: pass

    # ── Agent Chain ───────────────────────────────────────────

    async def chain(self, agent_type: str, prompt: str, db=None, context=None, **opts) -> LlmResponse:
        from app.services.agents.hub import get_agent_hub
        hub = get_agent_hub()
        cfg = _cfg("agent_chains", default={}).get(agent_type)
        if not cfg: return await hub.run(agent_type, prompt, db=db, context=context, **opts)
        if cfg.get("type") == "review":
            from app.services.llm_service import get_llm_service
            primary = await hub.run(agent_type, prompt, db=db, context=context, **opts)
            if not primary.content: return primary
            review_input = f"{cfg.get('reviewer_prompt', 'Review output quality.')}\n\n--- OUTPUT ---\n{primary.content[:3000]}"
            review_resp = await get_llm_service().complete(prompt=review_input, system_prompt="You are a quality reviewer. JSON only.")
            try:
                review = json.loads(review_resp.content)
                approved, score, feedback = review.get("approved", True), review.get("score", 8), review.get("feedback", "")
            except (json.JSONDecodeError, TypeError):
                approved, score, feedback = True, 7, ""
            if approved:
                primary.metadata = primary.metadata or {}; primary.metadata["review_score"] = score; return primary
            retry = await hub.run(agent_type, f"{prompt}\n\n--- FEEDBACK ---\n{feedback}", db=db, context=context, **opts)
            retry.metadata = {"review_score": score, "chain_retry": True}; return retry
        return await hub.run(agent_type, prompt, db=db, context=context, **opts)

    # ── Prompt Optimization (OPRO) ────────────────────────────

    async def select_prompt(self, agent_type: str, db: AsyncSession) -> Tuple[str, Optional[int]]:
        import random
        if random.random() < get_settings().ai_explore_rate:
            c = await self._get_prompt(db, agent_type, "candidate")
            if c: return c.prompt_text, c.id
        champ = await self._get_prompt(db, agent_type, "champion")
        if champ: return champ.prompt_text, champ.id
        active = await self._get_prompt(db, agent_type, None, is_active=True)
        return (active.prompt_text, active.id) if active else (None, None)

    async def _get_prompt(self, db, agent_type, status, is_active=False):
        conds = [AgentOptimization.agent_type == agent_type]
        if status: conds.append(AgentOptimization.status == status)
        if is_active: conds.append(AgentOptimization.is_active == True)
        result = await db.execute(select(AgentOptimization).where(and_(*conds)).order_by(AgentOptimization.version.desc()).limit(1))
        return result.scalar_one_or_none()

    async def record_result(self, db, opt_id: int, quality_score: float, passed: bool):
        try:
            opt = (await db.execute(select(AgentOptimization).where(AgentOptimization.id == opt_id))).scalar_one_or_none()
            if not opt: return
            opt.record_trial(quality_score, passed); await db.commit()
            if opt.status == "candidate" and opt.trial_count >= _cfg("prompt_engine", "min_trials_for_promotion", 10):
                await self._check_promotion(db, opt)
        except Exception as e:
            logger.warning(f"Brain.record_result: {e}")

    async def _check_promotion(self, db, candidate):
        pe = _cfg("prompt_engine", default={})
        margin, min_wr = pe.get("promotion_margin", 0.5), pe.get("min_win_rate", 60.0)
        champ = await self._get_prompt(db, candidate.agent_type, "champion")
        if not champ:
            candidate.status = "champion"; candidate.is_active = True; await db.commit(); return
        if candidate.avg_score > (champ.avg_score + margin) and candidate.win_rate >= min_wr:
            champ.status = "retired"; champ.is_active = False
            candidate.status = "champion"; candidate.is_active = True; await db.commit()
            logger.info(f"Brain: PROMOTED {candidate.agent_type} v{candidate.version}")
        elif candidate.trial_count >= pe.get("min_trials_for_promotion", 10) * pe.get("retirement_trials_multiplier", 2) and candidate.win_rate < pe.get("retirement_win_rate_threshold", 30.0):
            candidate.status = "retired"; await db.commit()

    async def bootstrap_prompt(self, db, agent_type: str, yaml_prompt: str):
        existing = await self._get_prompt(db, agent_type, "champion")
        if existing: return existing
        opt = AgentOptimization(agent_type=agent_type, version=0, prompt_text=yaml_prompt, notes="Bootstrapped", is_active=True, status="champion")
        db.add(opt); await db.commit(); await db.refresh(opt); return opt

    async def run_optimization_cycle(self, db) -> dict:
        import importlib
        results = {"optimized": 0, "skipped": 0, "errors": []}
        try:
            hub = (importlib.import_module("app.services.agents.hub")).get_agent_hub()
            for agent in [a["identifier"] for a in hub.agent_info()]:
                champ = await self._get_prompt(db, agent, "champion")
                if not champ: continue
                if await self._get_prompt(db, agent, "candidate"): results["skipped"] += 1; continue
                if champ.win_rate < 70.0 and champ.trial_count >= _cfg("prompt_engine", "min_trials_for_promotion", 10):
                    new_prompt = await self._meta_improve(agent, champ)
                    if new_prompt:
                        db.add(AgentOptimization(agent_type=agent, version=champ.version+1, prompt_text=new_prompt,
                                                 notes=f"OPRO from v{champ.version}", status="candidate", is_active=False))
                        results["optimized"] += 1
            await db.commit()
        except Exception as e:
            results["errors"].append(str(e))
        return results

    async def _meta_improve(self, agent_type: str, current) -> Optional[str]:
        from app.services.llm_service import get_llm_service
        try:
            resp = await get_llm_service().complete(
                prompt=f"AGENT: {agent_type}\nCURRENT PROMPT:\n{current.prompt_text}\nWIN RATE: {current.win_rate:.1f}%\nGenerate improved version. Output ONLY the new prompt.",
                system_prompt="You are a Prompt Engineer. Improve this AI agent prompt.",
                driver="groq", model=_cfg("fallback_models", "meta_optimizer", default="llama-3.1-8b-instant"))
            return resp.content.strip() if resp.content else None
        except Exception: return None

    # ── Response Filtering ────────────────────────────────────

    def filter_response(self, raw_content: str, agent_config: dict | None = None) -> AgentResult:
        return self._filter.filter(raw_content, agent_config)

    # ── Speculative Execution ─────────────────────────────────

    async def execute_speculative(self, prompt: str, system_prompt: str, agent_type: str,
                                   expected_fields: list[str] | None = None, **options) -> LlmResponse:
        from app.services.intelligence.guardian import get_guardian
        cfg = _cfg("speculative", default={})
        if not cfg.get("enabled", False):
            return await self.execute(prompt, system_prompt, agent_type, expected_fields=expected_fields, **options)
        try:
            draft = await asyncio.wait_for(
                self._call_driver(prompt, system_prompt, driver=cfg.get("draft_driver", "groq"),
                                  model=cfg.get("draft_model"), max_tokens=cfg.get("draft_max_tokens", 1024), **options),
                timeout=cfg.get("draft_timeout_s", 10))
        except Exception:
            return await self.execute(prompt, system_prompt, agent_type, expected_fields=expected_fields, **options)
        score = get_guardian().score_response(draft, expected_fields)
        if score["total"] >= cfg.get("min_quality_score", 6.5):
            draft.metadata = {"speculative": "accepted", "score": score["total"]}
            await get_guardian().record_quality(agent_type, score["total"]); return draft
        final = await self._call_cloud(prompt, system_prompt, agent_type=agent_type, **options)
        final.metadata = {"speculative": "escalated"}; return final

    # ── Swarm (multi-agent decomposition) ─────────────────────

    async def execute_swarm(self, prompt: str, agent_type: str, db=None, **opts) -> LlmResponse:
        cfg = _cfg("swarm", default={})
        if not cfg.get("enabled", False):
            return await self.chain(agent_type, prompt, db=db, **opts)
        scout = await self._assess_complexity(prompt, agent_type)
        if scout["complexity"] not in cfg.get("trigger_complexities", ["complex"]):
            return await self.chain(agent_type, prompt, db=db, **opts)

        language = detect_language(prompt)
        # Decompose
        from app.services.llm_service import get_llm_service
        try:
            decompose_resp = await asyncio.wait_for(get_llm_service().complete(
                prompt=f"Decompose into {cfg.get('max_subtasks', 6)} or fewer sub-tasks:\n{prompt}\nReturn JSON: [{{\"agent\":\"id\",\"prompt\":\"task\"}}]",
                system_prompt="Task decomposition engine. JSON only.",
                driver=cfg.get("decompose_driver", "groq"), max_tokens=512), timeout=cfg.get("decompose_timeout_s", 15))
            subtasks = json.loads(decompose_resp.content.strip().removeprefix("```json").removesuffix("```").strip())
            if not isinstance(subtasks, list) or len(subtasks) < 2: raise ValueError("too few")
        except Exception:
            return await self.chain(agent_type, prompt, db=db, **opts)

        # Orchestrate
        sem = asyncio.Semaphore(cfg.get("max_concurrent", 6))
        async def _run(st):
            async with sem:
                try:
                    from app.services.agents.hub import get_agent_hub
                    sub = st.get("prompt", prompt)
                    if language not in ("english", "hinglish"): sub = f"[Respond in {language}]\n{sub}"
                    r = await get_agent_hub().run(st.get("agent", agent_type), sub, db=db, **opts)
                    return {"agent": st.get("agent"), "content": r.content, "tokens": r.total_tokens}
                except Exception: return {"agent": st.get("agent", "?"), "content": ""}

        results = await asyncio.gather(*[_run(s) for s in subtasks])
        valid = [r for r in results if r.get("content")]
        if not valid: return await self.chain(agent_type, prompt, db=db, **opts)

        # Synthesize
        ctx = "\n\n---\n\n".join(f"[{r['agent']}]:\n{r['content'][:2000]}" for r in valid)
        lang_hint = f" Respond in {language}." if language not in ("english", "hinglish") else ""
        try:
            resp = await get_llm_service().complete(
                prompt=f"Synthesize for: \"{prompt}\"\n\nAGENT OUTPUTS:\n{ctx}\n\nMerge into one coherent response.{lang_hint}",
                system_prompt="Synthesis engine.", driver=cfg.get("synthesize_driver", "groq"),
                max_tokens=cfg.get("subtask_max_tokens", 2048))
            resp.metadata = {"swarm": True, "sub_agents": [r["agent"] for r in valid]}
            resp.total_tokens += sum(r.get("tokens", 0) for r in valid); return resp
        except Exception:
            return LlmResponse(content="\n\n".join(r["content"] for r in valid), total_tokens=0,
                               driver="swarm", model="concatenated", metadata={"swarm": True, "fallback": True})

    # ── Escalation ────────────────────────────────────────────

    async def escalate(self, brand_id: str, session_id: str, question: str, ai_response: str, confidence: float) -> bool:
        import httpx
        cfg = _cfg("chatbot", default={}).get("escalation", {})
        if confidence >= cfg.get("confidence_threshold", 0.6): return False
        wa = cfg.get("whatsapp", {})
        phone, api_url = wa.get("default_owner_phone"), wa.get("api_url")
        if not all([phone, api_url]): return False
        msg = f"🤖 *Query Alert*\n❓ {question}\n🤖 Answer ({confidence:.0%}):\n{ai_response[:300]}"
        try:
            async with httpx.AsyncClient(timeout=30) as c:
                r = await c.post(api_url, json={"phone": phone, "message": msg},
                                 headers={"Authorization": f"Bearer {wa.get('api_key', '')}"} if wa.get("api_key") else {})
                return r.status_code in (200, 201)
        except Exception as e:
            logger.error(f"Brain.escalate: {e}"); return False

    async def resolve_escalation(self, escalation_id: str, owner_answer: str) -> dict:
        try:
            from app.services.intelligence.chatbot_engine import get_chatbot_engine
            await get_chatbot_engine().learn_from_owner(brand_id="", question="", answer=owner_answer, session_id="")
            return {"status": "resolved", "id": escalation_id}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    # ── CoT Injection ─────────────────────────────────────────

    def _inject_thinking(self, system_prompt: str, complexity: str, agent_type: str) -> str:
        if not system_prompt or complexity == "simple": return system_prompt
        inject = complexity == "complex" or (complexity == "moderate" and agent_type in
                  {"seo", "email_campaign", "copywriter", "market_analyst", "coding_assistant"})
        return f"{system_prompt}\n{THINKING_INJECTION}" if inject else system_prompt


# ─── Singleton ────────────────────────────────────────────────

_brain: Brain | None = None
_brain_lock = threading.Lock()


def get_brain() -> Brain:
    global _brain
    if _brain is None:
        with _brain_lock:
            if _brain is None:
                _brain = Brain()
    return _brain
