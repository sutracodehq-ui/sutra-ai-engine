"""
Brain — Unified execution pipeline for the SutraCode AI Engine.

Software Factory Principle: One file for all execution logic.
Everything is config-driven via intelligence_config.yaml.

Absorbs: smart_router, hybrid_router, agent_chain, escalation_manager,
         response_filter, llm_queue, prompt_engine, meta_prompt_optimizer

Architecture:
    Request → Brain.route() → Brain.execute() → LlmResponse
                  ↓                  ↓
           YAML config         Guardian (safety)
           (routing rules)     Memory (cache/RAG)
                               Driver (LLM adapter)
"""

import asyncio
import hashlib
import json
import logging
import threading
import time
import unicodedata
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

# ─── Config Loader (cached, 60s TTL) ──────────────────────────

_config_cache: dict | None = None
_config_ts: float = 0
_CONFIG_TTL: float = 60.0

# Precomputed lookup sets (rebuilt on config reload)
_indic_scripts: frozenset | None = None
_hinglish_words: frozenset | None = None
_complex_signals: frozenset | None = None
_simple_signals: frozenset | None = None


def _load_config() -> dict:
    """Load full intelligence config. Cached with 60s TTL."""
    global _config_cache, _config_ts, _indic_scripts, _hinglish_words, _complex_signals, _simple_signals
    now = time.monotonic()
    if _config_cache is not None and (now - _config_ts) < _CONFIG_TTL:
        return _config_cache

    path = Path("intelligence_config.yaml")
    if not path.exists():
        _config_cache = {}
        _config_ts = now
        return _config_cache

    with open(path) as f:
        _config_cache = yaml.safe_load(f) or {}
    _config_ts = now

    # Reset precomputed sets
    _indic_scripts = _hinglish_words = _complex_signals = _simple_signals = None
    return _config_cache


def _cfg(section: str, key: str = None, default=None):
    """Read config value: _cfg('smart_router', 'sample_chars', 100)."""
    data = _load_config().get(section, {})
    if key is None:
        return data or default
    return data.get(key, default)


def _ensure_sets():
    """Build frozensets from YAML lists on first use."""
    global _indic_scripts, _hinglish_words, _complex_signals, _simple_signals
    if _indic_scripts is not None:
        return
    sr = _cfg("smart_router", default={})
    _indic_scripts = frozenset(sr.get("indic_scripts", []))
    _hinglish_words = frozenset(sr.get("hinglish_words", []))
    _complex_signals = frozenset(sr.get("complex_signals", []))
    _simple_signals = frozenset(sr.get("simple_signals", []))


# ─── Smart Routing (O(1) Decision Tree) ───────────────────────

def detect_language(text: str) -> str:
    """O(1) language detection: 'indic', 'hinglish', or 'english'."""
    if not text:
        return "english"
    _ensure_sets()
    sr = _cfg("smart_router", default={})
    sample = text[:sr.get("sample_chars", 100)]
    alpha_count = indic_count = 0
    for char in sample:
        if not char.isalpha():
            continue
        alpha_count += 1
        try:
            name = unicodedata.name(char, "")
            for script in _indic_scripts:
                if name.startswith(script):
                    indic_count += 1
                    break
        except ValueError:
            pass
    if alpha_count > 0 and (indic_count / alpha_count) > sr.get("indic_threshold", 0.3):
        return "indic"
    words = text.lower().split()[:sr.get("sample_words", 10)]
    if sum(1 for w in words if w in _hinglish_words) >= sr.get("hinglish_min_matches", 2):
        return "hinglish"
    return "english"


# ─── Response Filter ──────────────────────────────────────────

class _ResponseFilter:
    """Normalizes raw LLM text → AgentResult."""

    STANDARD_FIELDS = {"suggestions"}

    def filter(self, raw: str, agent_config: dict | None = None) -> AgentResult:
        if not raw or not raw.strip():
            return AgentResult(data={"content": ""}, suggestions=[], raw=raw or "", parsed=False)
        parsed, ok = self._parse_json(raw)
        if not ok or not isinstance(parsed, dict):
            return AgentResult(data={"content": raw}, suggestions=[], raw=raw, parsed=False)
        suggestions = self._extract_suggestions(parsed)
        if agent_config:
            self._validate_schema(parsed, agent_config)
        return AgentResult(data=parsed, suggestions=suggestions, raw=raw, parsed=True)

    def _parse_json(self, raw: str) -> tuple[Any, bool]:
        text = raw.strip()
        try:
            return json.loads(text), True
        except json.JSONDecodeError:
            pass
        cleaned = self._strip_fences(text)
        if cleaned != text:
            try:
                return json.loads(cleaned), True
            except json.JSONDecodeError:
                pass
        obj = self._extract_json_object(text)
        if obj:
            try:
                return json.loads(obj), True
            except json.JSONDecodeError:
                pass
        return None, False

    def _strip_fences(self, text: str) -> str:
        import re
        m = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
        return m.group(1).strip() if m else text

    def _extract_json_object(self, text: str) -> str | None:
        start = text.find("{")
        if start == -1:
            return None
        depth = 0
        in_str = esc = False
        for i in range(start, len(text)):
            ch = text[i]
            if esc:
                esc = False
                continue
            if ch == "\\":
                esc = True
                continue
            if ch == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
        return None

    def _extract_suggestions(self, data: dict) -> list[str]:
        raw = data.pop("suggestions", None)
        if raw is None:
            return []
        if isinstance(raw, list):
            return [str(s) for s in raw if s]
        if isinstance(raw, str) and raw.strip():
            return [raw.strip()]
        return []

    def _validate_schema(self, data: dict, config: dict):
        schema = config.get("response_schema", {})
        fields = schema.get("fields", []) if isinstance(schema, dict) else schema if isinstance(schema, list) else []
        fields = [f for f in fields if f not in self.STANDARD_FIELDS]
        if fields:
            missing = [f for f in fields if f not in data]
            if missing:
                logger.warning(f"Brain.filter: response missing fields: {missing}")


# ─── LLM Queue (Semaphore-based concurrency) ─────────────────

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
                self._waiting -= 1
                return
            try:
                await asyncio.wait_for(self._sem.acquire(), timeout=2.0)
                break
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                self._waiting -= 1
                return
        else:
            await self._sem.acquire()
        self._waiting -= 1
        self._active += 1
        cancelled = False
        try:
            yield _sse({"type": "status", "stage": "calculating"})
            gen = gen_fn()
            count = 0
            try:
                async for chunk in gen:
                    count += 1
                    if request and count % 5 == 0 and await request.is_disconnected():
                        cancelled = True
                        break
                    yield chunk
            finally:
                if cancelled:
                    await gen.aclose()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Brain.queue: stream error: {e}")
            yield _sse({"type": "error", "message": str(e)})
        finally:
            self._active -= 1
            self._sem.release()


# ─── Brain: The Unified Execution Pipeline ────────────────────

class Brain:
    """
    Config-driven execution pipeline.

    Responsibilities:
    1. route()         — O(1) driver + model selection (from smart_router)
    2. execute()       — local-first → quality gate → cloud (from hybrid_router)
    3. chain()         — multi-agent review chains (from agent_chain)
    4. select_prompt() — OPRO champion/candidate selection (from prompt_engine)
    5. filter()        — response normalization (from response_filter)
    6. queue           — semaphore-based concurrency (from llm_queue)

    All thresholds, weights, and routing rules come from YAML.
    Multi-agent flexibility is PRESERVED: SutraAgent + hub.py are untouched.
    """

    def __init__(self):
        self._filter = _ResponseFilter()
        self._queue = _LlmQueue()

    @property
    def queue(self) -> _LlmQueue:
        return self._queue

    # ─── Routing (absorbs smart_router + hybrid_router) ──────

    def route(self, prompt: str, agent_type: str, circuit_breaker=None) -> dict:
        """O(1) routing decision: language → complexity → driver → model."""
        settings = get_settings()
        if not settings.ai_smart_router_enabled:
            return {"driver": settings.ai_driver, "model": None, "complexity": "moderate",
                    "language": "english", "reason": "smart router disabled"}

        language = detect_language(prompt)
        complexity = self._assess_complexity(prompt, agent_type)
        driver = self._pick_driver(complexity, language, circuit_breaker)
        model = self._pick_model(driver, complexity)

        parts = []
        if language != "english":
            parts.append(f"{language} detected")
        parts.append(f"{complexity} task → {driver}")
        if model:
            parts.append(f"({model})")
        reason = " ".join(parts)
        logger.info(f"Brain.route: {reason}")

        return {"driver": driver, "model": model, "complexity": complexity,
                "language": language, "reason": reason}

    def _assess_complexity(self, prompt: str, agent_type: str) -> str:
        _ensure_sets()
        sr = _cfg("smart_router", default={})
        wc = len(prompt.split(None, 100))
        buckets = sr.get("length_buckets", {"short": 10, "long": 80})
        bucket = "short" if wc < buckets.get("short", 10) else ("long" if wc > buckets.get("long", 80) else "medium")

        n = sr.get("signal_words", 3)
        words = prompt.lower().split(None, n + 1)[:n]
        signal = "none"
        for w in words:
            if w in _complex_signals:
                signal = "complex"
                break
        if signal == "none":
            for w in words:
                if w in _simple_signals:
                    signal = "simple"
                    break

        agent_tier = "moderate"
        try:
            from app.services.agents.hub import get_agent_hub
            agent_tier = get_agent_hub().get(agent_type)._config.get("complexity", "moderate")
        except Exception:
            pass

        key = f"{bucket}_{signal}_{agent_tier}"
        return sr.get("decision_table", {}).get(key, sr.get("default_complexity", "moderate"))

    def _pick_driver(self, complexity: str, language: str, cb=None) -> str:
        sr = _cfg("smart_router", default={})
        lang_key = "indic" if language in ("indic", "hinglish") else "english"
        chains = sr.get("driver_chains", {})
        chain = chains.get(lang_key, {}).get(complexity, ["ollama", "groq"])
        settings = get_settings()
        key_map = {
            "openai": settings.openai_api_key, "anthropic": settings.anthropic_api_key,
            "gemini": settings.gemini_api_key, "groq": settings.groq_api_key,
            "sarvam": settings.sarvam_api_key, "nvidia": settings.nvidia_api_key,
            "ollama": "always",
        }
        for d in chain:
            if not key_map.get(d, ""):
                continue
            if cb and not cb.is_available(d):
                continue
            return d
        return settings.ai_driver

    def _pick_model(self, driver: str, complexity: str) -> Optional[str]:
        sr = _cfg("smart_router", default={})
        model = sr.get("model_tiers", {}).get(driver, {}).get(complexity)
        if model is None:
            s = get_settings()
            model = {"ollama": s.ollama_model, "sarvam": s.sarvam_model, "nvidia": s.nvidia_model}.get(driver)
        return model

    def select_model(self, prompt: str, agent_type: str, driver: str) -> Optional[str]:
        complexity = self._assess_complexity(prompt, agent_type)
        return self._pick_model(driver, complexity)

    # ─── Execution (absorbs hybrid_router.execute) ───────────

    async def execute(
        self, prompt: str, system_prompt: str, agent_type: str,
        expected_fields: list[str] | None = None, **options,
    ) -> LlmResponse:
        """Local-first → quality gate → cloud escalation."""
        from app.services.intelligence.guardian import get_guardian

        settings = get_settings()
        guardian = get_guardian()

        if not settings.ai_hybrid_routing:
            return await self._call_driver(prompt, system_prompt, **options)

        route_hint = await guardian.get_route_hint(agent_type)
        logger.info(f"Brain.execute: agent={agent_type}, hint={route_hint}")

        if route_hint == "direct_cloud":
            resp = await self._call_cloud(prompt, system_prompt, agent_type=agent_type, **options)
            await self._auto_train(agent_type, prompt, resp)
            return resp

        start = time.monotonic()
        local_resp = await self._call_local(prompt, system_prompt, **options)
        latency = round((time.monotonic() - start) * 1000)

        if route_hint == "fast_local":
            logger.info(f"Brain.execute: FAST_LOCAL {agent_type} ({latency}ms)")
            await guardian.record_quality(agent_type, 8.0)
            return local_resp

        score_result = guardian.score_response(local_resp, expected_fields)
        score = score_result["total"]

        if score_result["passed"]:
            await guardian.record_quality(agent_type, score)
            logger.info(f"Brain.execute: LOCAL_PASS {agent_type} ({latency}ms, score={score})")
            return local_resp

        logger.info(f"Brain.execute: ESCALATING {agent_type} (score={score})")
        cloud_resp = await self._call_cloud(prompt, system_prompt, agent_type=agent_type, **options)
        cloud_score = guardian.score_response(cloud_resp, expected_fields)["total"]
        await guardian.record_quality(agent_type, score)

        if cloud_score >= score_result["threshold"]:
            await self._auto_train(agent_type, prompt, cloud_resp)
            return cloud_resp

        return cloud_resp if cloud_score > score else local_resp

    async def _call_local(self, prompt: str, system_prompt: str, **opts) -> LlmResponse:
        from app.services.llm_service import get_llm_service
        try:
            return await get_llm_service().complete(prompt=prompt, system_prompt=system_prompt, driver="ollama", **opts)
        except Exception as e:
            logger.warning(f"Brain: local failed: {e}")
            return LlmResponse(content="", total_tokens=0, driver="ollama", model="qwen2.5:3b", metadata={"error": str(e)})

    async def _call_cloud(self, prompt: str, system_prompt: str, **opts) -> LlmResponse:
        from app.services.llm_service import get_llm_service
        svc = get_llm_service()
        agent_type = opts.pop("agent_type", "unknown")
        model_overrides = {}
        settings = get_settings()
        if settings.ai_smart_router_enabled:
            try:
                for drv in ["groq", "gemini", "anthropic"]:
                    m = self.select_model(prompt, agent_type, drv)
                    if m:
                        model_overrides[drv] = m
            except Exception:
                pass
        tiers = [("groq", "Tier1:Free"), ("gemini", "Tier2:Smart"), ("anthropic", "Tier3:Premium")]
        for driver, label in tiers:
            try:
                mo = model_overrides.get(driver)
                resp = await svc.complete(prompt=prompt, system_prompt=system_prompt, driver=driver, model=mo, **opts)
                resp.metadata = resp.metadata or {}
                resp.metadata["cloud_tier"] = label
                return resp
            except Exception as e:
                logger.warning(f"Brain: {label} ({driver}) failed: {e}")
        return LlmResponse(content="", total_tokens=0, driver="none", model="none", metadata={"error": "all cloud failed"})

    async def _call_driver(self, prompt: str, system_prompt: str, **opts) -> LlmResponse:
        from app.services.llm_service import get_llm_service
        return await get_llm_service().complete(prompt=prompt, system_prompt=system_prompt, **opts)

    async def _auto_train(self, agent_type: str, prompt: str, resp: LlmResponse):
        if not resp.content:
            return
        try:
            from app.services.intelligence.memory import get_memory
            mem = get_memory()
            await mem.remember(agent_type, prompt, resp.content, quality_score=1.0)
        except Exception as e:
            logger.warning(f"Brain: auto-train failed: {e}")

    # ─── Agent Chain (absorbs agent_chain) ───────────────────

    async def chain(self, agent_type: str, prompt: str, db=None, context=None, **opts) -> LlmResponse:
        """Execute → review chain if configured in YAML."""
        from app.services.agents.hub import get_agent_hub
        hub = get_agent_hub()
        chains = _cfg("agent_chains", default={})
        chain_cfg = chains.get(agent_type)
        if not chain_cfg:
            return await hub.run(agent_type, prompt, db=db, context=context, **opts)

        if chain_cfg.get("type") == "review":
            return await self._review_chain(hub, agent_type, prompt, chain_cfg, db, context, **opts)
        return await hub.run(agent_type, prompt, db=db, context=context, **opts)

    async def _review_chain(self, hub, agent_type, prompt, cfg, db=None, context=None, **opts) -> LlmResponse:
        from app.services.llm_service import get_llm_service
        primary = await hub.run(agent_type, prompt, db=db, context=context, **opts)
        if not primary.content:
            return primary

        review_prompt = cfg.get("reviewer_prompt", "Review the following output for quality.")
        review_input = f"{review_prompt}\n\n--- AGENT OUTPUT ---\n{primary.content[:3000]}"
        review_resp = await get_llm_service().complete(prompt=review_input, system_prompt="You are a quality reviewer. Respond with JSON only.", driver="ollama")

        try:
            review = json.loads(review_resp.content)
            approved, score, feedback = review.get("approved", True), review.get("score", 8), review.get("feedback", "")
        except (json.JSONDecodeError, TypeError):
            approved, score, feedback = True, 7, ""

        if approved:
            primary.metadata = primary.metadata or {}
            primary.metadata["review_score"] = score
            return primary

        enhanced = f"{prompt}\n\n--- REVIEWER FEEDBACK ---\n{feedback}"
        retry = await hub.run(agent_type, enhanced, db=db, context=context, **opts)
        retry.metadata = retry.metadata or {}
        retry.metadata.update({"review_score": score, "chain_retry": True})
        return retry

    # ─── Prompt Optimization (absorbs prompt_engine) ─────────

    async def select_prompt(self, agent_type: str, db: AsyncSession) -> Tuple[str, Optional[int]]:
        """OPRO: select champion or candidate prompt for A/B testing."""
        import random
        explore_rate = get_settings().ai_explore_rate
        if random.random() < explore_rate:
            candidate = await self._get_prompt(db, agent_type, "candidate")
            if candidate:
                return candidate.prompt_text, candidate.id

        champion = await self._get_prompt(db, agent_type, "champion")
        if champion:
            return champion.prompt_text, champion.id

        active = await self._get_prompt(db, agent_type, None, is_active=True)
        if active:
            return active.prompt_text, active.id

        return None, None

    async def _get_prompt(self, db: AsyncSession, agent_type: str, status: str | None, is_active: bool = False) -> Optional[AgentOptimization]:
        conditions = [AgentOptimization.agent_type == agent_type]
        if status:
            conditions.append(AgentOptimization.status == status)
        if is_active:
            conditions.append(AgentOptimization.is_active == True)
        stmt = select(AgentOptimization).where(and_(*conditions)).order_by(AgentOptimization.version.desc()).limit(1)
        result = await db.execute(stmt)
        return result.scalar_one_or_none()

    async def record_result(self, db: AsyncSession, opt_id: int, quality_score: float, passed: bool):
        """Record trial and check for auto-promotion."""
        try:
            stmt = select(AgentOptimization).where(AgentOptimization.id == opt_id)
            result = await db.execute(stmt)
            opt = result.scalar_one_or_none()
            if not opt:
                return
            opt.record_trial(quality_score, passed)
            await db.commit()
            pe = _cfg("prompt_engine", default={})
            min_trials = pe.get("min_trials_for_promotion", 10)
            if opt.status == "candidate" and opt.trial_count >= min_trials:
                await self._check_promotion(db, opt)
        except Exception as e:
            logger.warning(f"Brain.record_result: {e}")

    async def _check_promotion(self, db: AsyncSession, candidate: AgentOptimization):
        pe = _cfg("prompt_engine", default={})
        margin = pe.get("promotion_margin", 0.5)
        min_wr = pe.get("min_win_rate", 60.0)
        retire_mult = pe.get("retirement_trials_multiplier", 2)
        retire_thr = pe.get("retirement_win_rate_threshold", 30.0)
        min_trials = pe.get("min_trials_for_promotion", 10)

        champion = await self._get_prompt(db, candidate.agent_type, "champion")
        if not champion:
            candidate.status = "champion"
            candidate.is_active = True
            await db.commit()
            return

        if candidate.avg_score > (champion.avg_score + margin) and candidate.win_rate >= min_wr:
            champion.status = "retired"
            champion.is_active = False
            candidate.status = "champion"
            candidate.is_active = True
            await db.commit()
            logger.info(f"Brain: PROMOTED {candidate.agent_type} v{candidate.version}")
        elif candidate.trial_count >= min_trials * retire_mult and candidate.win_rate < retire_thr:
            candidate.status = "retired"
            await db.commit()
            logger.info(f"Brain: RETIRED weak candidate {candidate.agent_type} v{candidate.version}")

    async def bootstrap_prompt(self, db: AsyncSession, agent_type: str, yaml_prompt: str) -> AgentOptimization:
        """Seed DB with YAML baseline as initial champion."""
        existing = await self._get_prompt(db, agent_type, "champion")
        if existing:
            return existing
        opt = AgentOptimization(agent_type=agent_type, version=0, prompt_text=yaml_prompt, notes="Bootstrapped from YAML", is_active=True, status="champion")
        db.add(opt)
        await db.commit()
        await db.refresh(opt)
        return opt

    # ─── OPRO: Prompt Optimization Cycle ─────────────────────

    async def run_optimization_cycle(self, db: AsyncSession) -> dict:
        """Analyze low-performing agents and generate improved candidates."""
        from sqlalchemy import func
        from app.models.ai_task import AiTask

        # 1. Identify agents with low performance (< 6.0 avg)
        # We look at the last 50 tasks for each agent
        results = {"optimized": 0, "skipped": 0, "errors": []}
        
        try:
            hub = (await (importlib.import_module("app.services.agents.hub"))).get_agent_hub()
            agents = [a["identifier"] for a in hub.agent_info()]

            for agent in agents:
                champion = await self._get_prompt(db, agent, "champion")
                if not champion: continue

                # Check if we already have a pending candidate
                candidate = await self._get_prompt(db, agent, "candidate")
                if candidate:
                    results["skipped"] += 1
                    continue

                # Analyze recent performance
                # (Conceptual query: select avg(quality) from tasks where agent=agent)
                # For now, we use a simpler heuristic: if champion win rate < 70%
                if champion.win_rate < 70.0 and champion.trial_count >= _cfg("prompt_engine", "min_trials_for_promotion", 10):
                    # Generate new version
                    new_prompt = await self._generate_improved_prompt(agent, champion)
                    if new_prompt:
                        opt = AgentOptimization(
                            agent_type=agent, version=champion.version + 1,
                            prompt_text=new_prompt, notes=f"OPRO optimization from v{champion.version}",
                            status="candidate", is_active=False
                        )
                        db.add(opt)
                        results["optimized"] += 1
            
            await db.commit()
        except Exception as e:
            logger.error(f"Brain.optimization_cycle: {e}")
            results["errors"].append(str(e))

        return results

    async def _generate_improved_prompt(self, agent_type: str, current: AgentOptimization) -> Optional[str]:
        """Call a high-reasoning meta-LLM to improve the prompt."""
        from app.services.llm_service import get_llm_service
        meta_prompt = (
            "You are a Prompt Engineer. Improve this AI agent system prompt based on poor performance.\n\n"
            f"AGENT: {agent_type}\n"
            f"CURRENT PROMPT:\n{current.prompt_text}\n\n"
            "WIN RATE: {current.win_rate:.1f}%\n"
            "AVG SCORE: {current.avg_score:.1f}\n\n"
            "TASK: Generate a more precise, robust version of this prompt. "
            "Address edge cases and clarify instructions. Output ONLY the new prompt text."
        )
        try:
            # Use Groq or Gemini for meta-optimization (high intelligence)
            resp = await get_llm_service().complete(
                prompt=meta_prompt, system_prompt="You are a Meta-Prompt Optimizer.",
                driver="groq", model="llama-3.3-70b-versatile"
            )
            return resp.content.strip() if resp.content else None
        except Exception as e:
            logger.warning(f"Brain: meta-optimization failed for {agent_type}: {e}")
            return None

    # ─── Response Filtering ──────────────────────────────────

    def filter_response(self, raw_content: str, agent_config: dict | None = None) -> AgentResult:
        """Normalize raw LLM text → AgentResult."""
        return self._filter.filter(raw_content, agent_config)

    # ─── Escalation (absorbs escalation_manager) ─────────────

    async def escalate(self, brand_id: str, session_id: str, question: str, ai_response: str, confidence: float) -> bool:
        """Escalate to brand owner when confidence is low."""
        import httpx
        cfg = _cfg("chatbot", default={}).get("escalation", {})
        if confidence >= cfg.get("confidence_threshold", 0.6):
            return False

        esc_id = hashlib.md5(f"{brand_id}:{session_id}:{question}:{datetime.now().isoformat()}".encode()).hexdigest()[:12]
        wa = cfg.get("whatsapp", {})
        phone, api_url = wa.get("default_owner_phone"), wa.get("api_url")
        if not all([phone, api_url]):
            return False

        msg = (f"🤖 *Customer Query Alert*\n\n❓ *Question:*\n{question}\n\n"
               f"🤖 *My answer ({confidence:.0%}):*\n{ai_response[:300]}\n\n"
               f"Reply with the correct answer.\n_Ref: {esc_id}_")
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                r = await client.post(api_url, json={"phone": phone, "message": msg, "escalation_id": esc_id},
                                       headers={"Authorization": f"Bearer {wa.get('api_key', '')}"} if wa.get("api_key") else {})
                return r.status_code in (200, 201)
        except Exception as e:
            logger.error(f"Brain.escalate: {e}")
            return False


# ─── Singleton ──────────────────────────────────────────────────

_brain: Brain | None = None
_brain_lock = threading.Lock()


def get_brain() -> Brain:
    global _brain
    if _brain is None:
        with _brain_lock:
            if _brain is None:
                _brain = Brain()
    return _brain
