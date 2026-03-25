"""
Guardian — Unified safety, quality, and resilience engine.

Software Factory Principle: One file for all protection logic.
Everything is config-driven via intelligence_config.yaml.

Absorbs: quality_engine, moderation, pii_redactor, competitor_lock,
         circuit_breaker, retry_strategy, rate_limiter, token_budget,
         token_forecaster, thinking, sentiment
"""

import asyncio
import logging
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, TypeVar

import httpx
import yaml

from app.config import get_settings
from app.services.drivers.base import LlmResponse

logger = logging.getLogger(__name__)
T = TypeVar("T")

# ─── Config Loader ─────────────────────────────────────────────

_cfg_cache: dict | None = None
_cfg_ts: float = 0


def _load_cfg() -> dict:
    global _cfg_cache, _cfg_ts
    now = time.monotonic()
    if _cfg_cache is not None and (now - _cfg_ts) < 60.0:
        return _cfg_cache
    path = Path("intelligence_config.yaml")
    _cfg_cache = yaml.safe_load(open(path)) if path.exists() else {}
    _cfg_ts = now
    return _cfg_cache


def _sec(section: str, default=None):
    return _load_cfg().get(section, default or {})


# ─── Circuit Breaker (absorbs circuit_breaker.py) ─────────────

class _CircuitBreaker:
    """Per-driver circuit breaker: CLOSED → OPEN → HALF_OPEN → CLOSED."""

    def __init__(self, threshold: int = 3, cooldown: int = 60):
        self._threshold = threshold
        self._cooldown = cooldown
        self._states: dict[str, dict] = {}

    def _state(self, d: str) -> dict:
        if d not in self._states:
            self._states[d] = {"state": "closed", "fails": 0, "last_fail": 0, "last_ok": 0}
        return self._states[d]

    def is_available(self, driver: str) -> bool:
        s = self._state(driver)
        if s["state"] == "closed":
            return True
        if s["state"] == "open" and (time.time() - s["last_fail"]) >= self._cooldown:
            s["state"] = "half_open"
            return True
        return s["state"] == "half_open"

    def record_success(self, driver: str):
        s = self._state(driver)
        was_ho = s["state"] == "half_open"
        s.update(state="closed", fails=0, last_ok=time.time())
        if was_ho:
            logger.info(f"Guardian.circuit: {driver} → CLOSED (test ok)")

    def record_failure(self, driver: str):
        s = self._state(driver)
        s["fails"] += 1
        s["last_fail"] = time.time()
        if s["state"] == "half_open":
            s["state"] = "open"
        elif s["fails"] >= self._threshold:
            s["state"] = "open"
            logger.warning(f"Guardian.circuit: {driver} → OPEN ({s['fails']} fails)")

    def status(self) -> dict[str, str]:
        return {d: s["state"] for d, s in self._states.items()}

    def reset(self, driver: str):
        self._states.pop(driver, None)


# ─── Retry Strategy (absorbs retry_strategy.py) ──────────────

RETRYABLE_ERRORS = (TimeoutError, ConnectionError, ConnectionRefusedError, ConnectionResetError)
RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}


async def _retry(func: Callable, *args, max_retries: int = 2, base_delay: float = 1.0, **kwargs) -> T:
    """Execute with exponential backoff + jitter."""
    last_err = None
    for attempt in range(max_retries + 1):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            last_err = e
            if attempt >= max_retries:
                raise
            retryable = isinstance(e, RETRYABLE_ERRORS)
            sc = getattr(e, "status_code", None) or getattr(e, "status", None)
            if sc and int(sc) in RETRYABLE_STATUS:
                retryable = True
            msg = str(e).lower()
            if "rate limit" in msg or "too many requests" in msg:
                retryable = True
            if not retryable:
                raise
            delay = min(base_delay * (2 ** attempt), 30.0)
            delay += random.uniform(-delay * 0.25, delay * 0.25)
            logger.warning(f"Guardian.retry: attempt {attempt + 1}, retrying in {delay:.1f}s: {e}")
            await asyncio.sleep(delay)
    raise last_err


# ─── Guardian: The Unified Protection Engine ──────────────────

class Guardian:
    """
    Config-driven safety, quality, and resilience engine.

    Modules (all toggled via YAML):
    1. Quality scoring + per-agent tracking (from quality_engine)
    2. Content moderation (from moderation)
    3. PII redaction (from pii_redactor)
    4. Circuit breaker (from circuit_breaker)
    5. Retry strategy (from retry_strategy)
    6. Rate limiting (from rate_limiter)
    7. Token budget (from token_budget)
    """

    def __init__(self):
        res = _sec("resilience", {})
        cb = res.get("circuit_breaker", {})
        self._circuit = _CircuitBreaker(
            threshold=cb.get("threshold", 3),
            cooldown=cb.get("cooldown_seconds", 60),
        )
        rt = res.get("retry", {})
        self._retry_max = rt.get("max_retries", 2)
        self._retry_delay = rt.get("base_delay", 1.0)

        q = _sec("quality", {})
        self._quality_enabled = q.get("enabled", True)
        self._quality_threshold = q.get("threshold", get_settings().ai_hybrid_quality_threshold)
        self._quality_weights = q.get("weights", {"format": 0.35, "completeness": 0.30,
                                                   "length": 0.15, "coherence": 0.20})

        # Pre-compile PII regex patterns from YAML config
        safety = _sec("safety", {})
        pii_cfg = safety.get("pii", {})
        raw_patterns = pii_cfg.get("patterns", {})
        self._pii_patterns = [re.compile(p) for p in raw_patterns.values()]
        detect_patterns = pii_cfg.get("detect_patterns", {})
        self._pii_detect_patterns = [re.compile(p) for p in detect_patterns.values()]

        # Pre-compile sentiment patterns from YAML config
        sent_cfg = safety.get("sentiment", _sec("sentiment", {}))
        pos_words = sent_cfg.get("positive_words", ["good", "great", "awesome"])
        neg_words = sent_cfg.get("negative_words", ["bad", "wrong", "broken"])
        self._sentiment_pos = re.compile(r"\b(" + "|".join(pos_words) + r")\b")
        self._sentiment_neg = re.compile(r"\b(" + "|".join(neg_words) + r")\b")

        # Pre-load budget and timeout config
        self._budget_cfg = _sec("budget", {})
        self._timeouts = _sec("timeouts", _sec("resilience", {}).get("timeouts", {}))

    @property
    def circuit_breaker(self) -> _CircuitBreaker:
        return self._circuit

    # ─── Quality Scoring (absorbs quality_engine scoring) ────

    def score_response(self, response: LlmResponse, expected_fields: list[str] | None = None) -> dict:
        """Multi-dimensional quality scoring (single split, reused word count)."""
        if not self._quality_enabled:
            return {"total": 10, "passed": True, "threshold": self._quality_threshold, "dimensions": {}}

        content = response.content or ""
        dims = {}
        w = self._quality_weights

        # Single split — reuse for all dimensions
        words = content.split()
        word_count = len(words)

        # Format check
        if expected_fields:
            try:
                import json
                data = json.loads(content.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip())
                present = sum(1 for f in expected_fields if f in data)
                dims["format"] = round(10 * present / len(expected_fields), 1)
            except Exception:
                dims["format"] = 2.0
        else:
            dims["format"] = 8.0 if len(content) > 20 else 3.0

        # Completeness (reuses word_count)
        if word_count > 100:
            dims["completeness"] = 9.0
        elif word_count > 40:
            dims["completeness"] = 7.0
        elif word_count > 10:
            dims["completeness"] = 5.0
        else:
            dims["completeness"] = 2.0

        # Length (reuses word_count)
        if 50 < word_count < 2000:
            dims["length"] = 8.0
        elif word_count > 10:
            dims["length"] = 5.0
        else:
            dims["length"] = 2.0

        # Coherence (count sentence-ending chars in one pass)
        sentence_count = content.count(".") + content.count("!") + content.count("?")
        dims["coherence"] = min(8.5, max(3.0, sentence_count * 1.5))

        total = round(sum(dims.get(d, 5) * w.get(d, 0.25) for d in w), 2)
        return {"total": total, "passed": total >= self._quality_threshold,
                "threshold": self._quality_threshold, "dimensions": dims}

    # ─── Quality Tracking (absorbs quality_engine tracking) ──

    async def record_quality(self, agent_type: str, score: float):
        """Record quality score in Redis rolling window."""
        try:
            from app.services.connectivity.webhooks import get_redis
            redis = get_redis()
            cfg = _sec("quality", {}).get("tracking", {})
            key = f"sutra:quality:{agent_type}"
            window = cfg.get("window_size", 20)
            await redis.lpush(key, str(score))
            await redis.ltrim(key, 0, window - 1)
            await redis.expire(key, cfg.get("ttl_days", 7) * 86400)
        except Exception as e:
            logger.debug(f"Guardian: quality tracking skipped: {e}")

    async def get_route_hint(self, agent_type: str) -> str:
        """Adaptive routing hint from quality history."""
        try:
            from app.services.connectivity.webhooks import get_redis
            redis = get_redis()
            cfg = _sec("quality", {}).get("tracking", {})
            scores = await redis.lrange(f"sutra:quality:{agent_type}", 0, -1)
            if not scores or len(scores) < 3:
                return "standard"
            avg = sum(float(s) for s in scores) / len(scores)
            if avg >= cfg.get("fast_path_threshold", 8.0):
                return "fast_local"
            if avg <= cfg.get("direct_cloud_threshold", 5.0):
                return "direct_cloud"
            return "standard"
        except Exception:
            return "standard"

    def generate_retry_prompt(self, original: str, quality: dict) -> str:
        dims = quality.get("dimensions", {})
        weak = [d for d, s in dims.items() if s < 6.0]
        hints = {"format": "Return valid JSON.", "completeness": "Be more detailed.",
                 "length": "Provide a thorough answer.", "coherence": "Use clear structure."}
        feedback = " ".join(hints.get(w, "") for w in weak)
        return f"{original}\n\n[QUALITY FEEDBACK: {feedback}]" if feedback else original

    # ─── Moderation (absorbs moderation.py) ───────────────────

    async def moderate(self, text: str) -> dict:
        """Content safety check via OpenAI Moderation API."""
        if not text:
            return {"flagged": False, "categories": [], "score": 0.0}
        settings = get_settings()
        if not settings.openai_api_key:
            return {"flagged": False, "categories": [], "score": 0.0}
        try:
            mod_timeout = self._timeouts.get("moderation_s", 5)
            async with httpx.AsyncClient(timeout=float(mod_timeout)) as client:
                resp = await client.post("https://api.openai.com/v1/moderations",
                    headers={"Content-Type": "application/json", "Authorization": f"Bearer {settings.openai_api_key}"},
                    json={"input": text})
                resp.raise_for_status()
                result = resp.json()["results"][0]
                return {
                    "flagged": result["flagged"],
                    "categories": [c for c, v in result["categories"].items() if v],
                    "score": max(result.get("category_scores", {}).values()) if result.get("category_scores") else 0.0,
                }
        except Exception as e:
            logger.error(f"Guardian.moderate: {e}")
            return {"flagged": False, "categories": [], "score": 0.0}

    # ─── PII Redaction (absorbs pii_redactor.py) ─────────────

    def redact_pii(self, text: str, placeholder: str = "[REDACTED]") -> str:
        """Mask emails, phones, credit cards using pre-compiled YAML patterns."""
        if not text:
            return text
        for pattern in self._pii_patterns:
            text = pattern.sub(placeholder, text)
        return text

    def contains_pii(self, text: str) -> bool:
        """Detect PII using pre-compiled YAML detect patterns."""
        return any(p.search(text) for p in self._pii_detect_patterns)

    # ─── Rate Limiting (absorbs rate_limiter.py) ─────────────

    async def check_rate_limit(self, tenant_id: int, tenant_config: dict | None = None) -> dict:
        """Sliding window rate limiter via Redis sorted sets."""
        try:
            from app.services.connectivity.webhooks import get_redis
            redis = get_redis()
        except Exception:
            return {"allowed": True, "remaining": -1, "limit": 60, "retry_after": None}

        res = _sec("resilience", {}).get("rate_limiter", {})
        rpm = (tenant_config or {}).get("rpm", res.get("default_rpm", 60))
        key = f"sutra:ratelimit:{tenant_id}:rpm"
        now = time.time()
        window = 60
        try:
            pipe = redis.pipeline()
            pipe.zremrangebyscore(key, 0, now - window)
            pipe.zcard(key)
            pipe.zadd(key, {f"{now}": now})
            pipe.expire(key, window + 1)
            results = await pipe.execute()
            count = results[1]
            if count >= rpm:
                oldest = await redis.zrange(key, 0, 0, withscores=True)
                retry_after = round(oldest[0][1] + window - now, 1) if oldest else 1.0
                return {"allowed": False, "remaining": 0, "limit": rpm, "retry_after": retry_after}
            return {"allowed": True, "remaining": max(0, rpm - count - 1), "limit": rpm, "retry_after": None}
        except Exception as e:
            logger.warning(f"Guardian.rate_limit: {e}")
            return {"allowed": True, "remaining": -1, "limit": rpm, "retry_after": None}

    # ─── Token Budget (absorbs token_budget.py) ──────────────

    async def check_budget(self, tenant_id: int, tenant_config: dict | None = None) -> dict:
        """Per-tenant monthly token budget enforcement (config-driven)."""
        default_limit = self._budget_cfg.get("default_monthly_tokens", 1_000_000)
        try:
            from app.services.connectivity.webhooks import get_redis
            redis = get_redis()
        except Exception:
            return {"allowed": True, "usage": 0, "limit": default_limit, "percentage": 0, "level": "ALLOW"}

        limit = (tenant_config or {}).get("monthly_token_limit", default_limit)
        month = datetime.now(timezone.utc).strftime("%Y-%m")
        key = f"sutra:budget:{tenant_id}:monthly:{month}"
        try:
            usage = int(await redis.get(f"{key}:tokens") or 0)
        except Exception:
            usage = 0
        pct = (usage / limit * 100) if limit > 0 else 0
        if pct >= 100:
            return {"allowed": False, "usage": usage, "limit": limit, "percentage": round(pct, 1), "level": "BLOCK"}
        if pct >= 80:
            logger.warning(f"Guardian: tenant={tenant_id} at {pct:.0f}% budget")
        return {"allowed": True, "usage": usage, "limit": limit, "percentage": round(pct, 1),
                "level": "WARN" if pct >= 80 else "ALLOW"}

    async def record_usage(self, tenant_id: int, tokens: int, model: str = "unknown",
                           prompt_tokens: int = 0, completion_tokens: int = 0):
        """Record token usage (model costs from YAML config)."""
        try:
            from app.services.connectivity.webhooks import get_redis
            redis = get_redis()
            model_costs = self._budget_cfg.get("model_costs", {})
            fallback = self._budget_cfg.get("fallback_cost", {"input": 0.001, "output": 0.002})
            costs = model_costs.get(model, fallback)
            cost = (prompt_tokens / 1000 * costs["input"]) + (completion_tokens / 1000 * costs["output"])
            month = datetime.now(timezone.utc).strftime("%Y-%m")
            key = f"sutra:budget:{tenant_id}:monthly:{month}"
            ttl_days = self._budget_cfg.get("redis_ttl_days", 60)
            ttl_s = ttl_days * 86400
            pipe = redis.pipeline()
            pipe.incrby(f"{key}:tokens", tokens)
            pipe.incrbyfloat(f"{key}:cost", cost)
            pipe.expire(f"{key}:tokens", ttl_s)
            pipe.expire(f"{key}:cost", ttl_s)
            await pipe.execute()
        except Exception as e:
            logger.warning(f"Guardian.record_usage: {e}")

    # ─── Sentiment Analysis (absorbs sentiment.py) ───────────

    async def analyze_sentiment(self, text: str) -> dict:
        """Extract sentiment using pre-compiled patterns."""
        if not text:
            return {"sentiment": "neutral", "score": 0.5, "emotions": []}
        try:
            lower = text.lower()
            pos = len(self._sentiment_pos.findall(lower))
            neg = len(self._sentiment_neg.findall(lower))
            score = 0.5 + (0.1 * (pos - neg))
            sentiment = "positive" if score > 0.6 else ("negative" if score < 0.4 else "neutral")
            return {"sentiment": sentiment, "score": round(min(1.0, max(0.0, score)), 2), "emotions": []}
        except Exception:
            return {"sentiment": "neutral", "score": 0.5, "emotions": []}

    # ─── Token Forecasting (absorbs token_forecaster.py) ───────

    def forecast_tokens(self, prompt: str, expected_output_len: int = 0) -> dict:
        """Estimate token usage and cost BEFORE calling the LLM (config-driven)."""
        fc = self._budget_cfg.get("forecast_defaults", {})
        multiplier = fc.get("input_multiplier", 1.3)
        if expected_output_len == 0:
            expected_output_len = fc.get("default_output_tokens", 500)
        input_tokens = len(prompt.split()) * multiplier
        total = input_tokens + expected_output_len
        default_model = fc.get("default_model", "gpt-4o-mini")
        model_costs = self._budget_cfg.get("model_costs", {})
        fallback = self._budget_cfg.get("fallback_cost", {"input": 0.001, "output": 0.002})
        costs = model_costs.get(default_model, fallback)
        cost = (input_tokens / 1000 * costs["input"]) + (expected_output_len / 1000 * costs["output"])
        return {
            "input_tokens": int(input_tokens),
            "output_tokens": expected_output_len,
            "total_tokens": int(total),
            "estimated_cost_usd": round(cost, 6),
        }

    # ─── Retry Execution ─────────────────────────────────────

    async def with_retry(self, func: Callable, *args, **kwargs):
        """Execute with exponential backoff + jitter."""
        return await _retry(func, *args, max_retries=self._retry_max, base_delay=self._retry_delay, **kwargs)


# ─── Singleton ──────────────────────────────────────────────────

_guardian: Guardian | None = None

def get_guardian() -> Guardian:
    global _guardian
    if _guardian is None:
        _guardian = Guardian()
    return _guardian
