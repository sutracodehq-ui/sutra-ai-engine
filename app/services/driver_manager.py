"""
Driver Manager — Software Factory registry for LLM providers.

Config-driven: reads `AI_DRIVER` and `AI_FALLBACK_DRIVER` from settings,
resolves driver instances by name, and provides automatic fallback chains.

Integrates CircuitBreaker (skip dead drivers), RetryStrategy (transient fault handling),
and LoadBalancer (per-driver concurrency + smart overflow).
"""

import asyncio
import logging
import random
import threading
import time
from pathlib import Path
from typing import Any, AsyncGenerator, Callable

import yaml

from app.config import get_settings
from app.services.drivers.base import LlmDriver, LlmResponse

logger = logging.getLogger(__name__)


# ─── Isolated Circuit Breaker (DriverManager-only) ──────────────
# NOT shared with Guardian — this prevents moderation/KB failures
# from contaminating driver availability state.

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
            logger.info(f"DriverManager.circuit: {driver} → CLOSED (recovered)")

    def record_failure(self, driver: str):
        s = self._state(driver)
        s["fails"] += 1
        s["last_fail"] = time.time()
        if s["state"] == "half_open":
            s["state"] = "open"
        elif s["fails"] >= self._threshold:
            s["state"] = "open"
            logger.warning(f"DriverManager.circuit: {driver} → OPEN ({s['fails']} fails)")

    def status(self) -> dict[str, str]:
        return {d: s["state"] for d, s in self._states.items()}

    def reset(self, driver: str):
        self._states.pop(driver, None)


# ─── Load Balancer (per-driver concurrency + smart overflow) ────
# Tracks active requests per driver and enables instant overflow
# to the next available driver when a driver hits its limit.

def _load_driver_limits() -> dict[str, int]:
    """Load per-driver concurrency limits from intelligence_config.yaml."""
    try:
        path = Path("intelligence_config.yaml")
        if path.exists():
            cfg = yaml.safe_load(open(path)) or {}
            return cfg.get("load_balancer", {}).get("driver_limits", {})
    except Exception:
        pass
    return {}


# Default concurrency limits per driver (conservative for local, generous for cloud)
_DEFAULT_DRIVER_LIMITS = {
    "ollama": 2,       # CPU-bound, matches OLLAMA_NUM_PARALLEL
    "groq": 20,        # Cloud API, virtually unlimited
    "sarvam": 10,      # Cloud API
    "openai": 20,      # Cloud API
    "anthropic": 10,   # Cloud API
    "gemini": 20,      # Cloud API
    "nvidia": 10,      # Cloud API
}


class _DriverLoadBalancer:
    """Per-driver concurrency tracker with smart overflow.

    Tracks:
    - Active requests per driver (via asyncio.Semaphore)
    - Average latency per driver (exponential moving average)
    - Error rate per driver (rolling window)

    When a driver's slots are full, `pick_first_available()` reorders the
    fallback chain to skip overloaded drivers and route to the next one.
    """

    def __init__(self):
        yaml_limits = _load_driver_limits()
        self._limits: dict[str, int] = {**_DEFAULT_DRIVER_LIMITS, **yaml_limits}
        self._semaphores: dict[str, asyncio.Semaphore] = {}
        self._active: dict[str, int] = {}
        self._lock = asyncio.Lock()

        # Health tracking
        self._latency: dict[str, float] = {}     # EMA of latency per driver
        self._error_count: dict[str, int] = {}    # Error count in current window
        self._request_count: dict[str, int] = {}  # Total requests in current window
        self._window_start: float = time.time()
        self._window_size: float = 300.0          # 5 minute rolling window

    def _get_semaphore(self, driver: str) -> asyncio.Semaphore:
        if driver not in self._semaphores:
            limit = self._limits.get(driver, 10)
            self._semaphores[driver] = asyncio.Semaphore(limit)
            self._active[driver] = 0
        return self._semaphores[driver]

    def has_capacity(self, driver: str) -> bool:
        """Check if a driver has available slots without acquiring."""
        sem = self._get_semaphore(driver)
        return sem._value > 0

    async def acquire(self, driver: str, timeout: float = 0.0) -> bool:
        """Try to acquire a slot for a driver. Returns False if full (non-blocking)."""
        sem = self._get_semaphore(driver)
        if timeout <= 0:
            # Non-blocking: check immediately
            if sem._value > 0:
                await sem.acquire()
                self._active[driver] = self._active.get(driver, 0) + 1
                return True
            return False
        else:
            try:
                await asyncio.wait_for(sem.acquire(), timeout=timeout)
                self._active[driver] = self._active.get(driver, 0) + 1
                return True
            except asyncio.TimeoutError:
                return False

    def release(self, driver: str):
        """Release a slot back to the pool."""
        sem = self._get_semaphore(driver)
        self._active[driver] = max(0, self._active.get(driver, 0) - 1)
        sem.release()

    def record_latency(self, driver: str, latency_ms: float):
        """Record response latency (exponential moving average)."""
        alpha = 0.3  # Weight for new observation
        old = self._latency.get(driver, latency_ms)
        self._latency[driver] = alpha * latency_ms + (1 - alpha) * old

    def record_request(self, driver: str, success: bool):
        """Record a completed request for error rate tracking."""
        self._maybe_reset_window()
        self._request_count[driver] = self._request_count.get(driver, 0) + 1
        if not success:
            self._error_count[driver] = self._error_count.get(driver, 0) + 1

    def _maybe_reset_window(self):
        """Reset counters if the rolling window has elapsed."""
        if time.time() - self._window_start > self._window_size:
            self._error_count.clear()
            self._request_count.clear()
            self._window_start = time.time()

    def error_rate(self, driver: str) -> float:
        """Get error rate for a driver (0.0 - 1.0)."""
        total = self._request_count.get(driver, 0)
        if total == 0:
            return 0.0
        return self._error_count.get(driver, 0) / total

    def pick_first_available(self, chain: list[str], circuit: _CircuitBreaker) -> list[str]:
        """Reorder chain: available drivers first, overloaded drivers last.

        Priority: has_capacity AND circuit_ok > circuit_ok but full > rest
        Within each group, sort by error rate (lowest first).
        """
        available = []
        full_but_ok = []
        unavailable = []

        for d in chain:
            if not circuit.is_available(d):
                unavailable.append(d)
            elif self.has_capacity(d):
                available.append(d)
            else:
                full_but_ok.append(d)

        # Sort available by error rate (healthiest first)
        available.sort(key=lambda d: self.error_rate(d))

        reordered = available + full_but_ok + unavailable
        if reordered != list(chain):
            logger.info(
                f"LoadBalancer: reordered chain {chain} → {reordered} "
                f"(active: {dict((d, self._active.get(d, 0)) for d in chain)})"
            )
        return reordered

    def stats(self) -> dict:
        """Get load balancer stats for monitoring."""
        return {
            driver: {
                "active": self._active.get(driver, 0),
                "limit": self._limits.get(driver, 10),
                "has_capacity": self.has_capacity(driver),
                "avg_latency_ms": round(self._latency.get(driver, 0), 1),
                "error_rate": round(self.error_rate(driver), 3),
            }
            for driver in set(list(self._active.keys()) + list(self._limits.keys()))
        }


# ─── Retry Strategy (transient errors only) ─────────────────────
# Only retries known-transient errors (rate limits, timeouts, server errors).
# Does NOT retry 400 Bad Request, 401 Unauthorized, etc.

# Exceptions that are retryable (transient)
_RETRYABLE_ERRORS = (
    TimeoutError,
    ConnectionError,
    ConnectionRefusedError,
    ConnectionResetError,
)

# HTTP status codes that should trigger retry
_RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}


def _is_retryable(error: Exception) -> bool:
    """Check if an error is transient and worth retrying."""
    if isinstance(error, _RETRYABLE_ERRORS):
        return True

    # Check httpx/openai-style status codes
    status = getattr(error, "status_code", None) or getattr(error, "status", None)
    if status and int(status) in _RETRYABLE_STATUS_CODES:
        return True

    # Check for rate limit patterns in error message
    err_str = str(error).lower()
    if "rate limit" in err_str or "429" in err_str or "too many" in err_str:
        return True

    return False


async def _retry_transient(callback: Callable, driver: LlmDriver, *, max_retries: int = 2, base_delay: float = 1.0):
    """Execute with exponential backoff, retrying ONLY transient errors."""
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            return await callback(driver)
        except Exception as e:
            last_error = e
            if attempt < max_retries and _is_retryable(e):
                delay = base_delay * (2 ** attempt) * (0.75 + random.random() * 0.5)
                logger.info(f"DriverManager: retry {attempt + 1}/{max_retries} for {driver.name()} after {delay:.1f}s ({e.__class__.__name__})")
                await asyncio.sleep(delay)
            else:
                raise
    raise last_error


class DriverManager:
    """
    Factory registry for LLM drivers.

    Resolves drivers by name from a config-driven map.
    Provides fallback chain execution — if primary fails, tries fallback automatically.
    Integrates CircuitBreaker (skip dead drivers), RetryStrategy (transient handling),
    and LoadBalancer (per-driver concurrency + smart overflow).
    """

    # ─── Driver Registry (Software Factory: config → class mapping) ──

    DRIVER_MAP: dict[str, str] = {
        "openai": "app.services.drivers.openai_driver.OpenAiDriver",
        "anthropic": "app.services.drivers.anthropic_driver.AnthropicDriver",
        "gemini": "app.services.drivers.gemini_driver.GeminiDriver",
        "groq": "app.services.drivers.groq_driver.GroqDriver",
        "ollama": "app.services.drivers.ollama_driver.OllamaDriver",
        "sarvam": "app.services.drivers.sarvam_driver.SarvamDriver",
        "nvidia": "app.services.drivers.nvidia_driver.NvidiaDriver",
        "mock": "app.services.drivers.mock_driver.MockDriver",
    }

    def __init__(self):
        self._instances: dict[str, LlmDriver] = {}
        self._circuit = _CircuitBreaker(threshold=3, cooldown=60)
        self._lb = _DriverLoadBalancer()

    @property
    def circuit_breaker(self) -> _CircuitBreaker:
        return self._circuit

    @property
    def load_balancer(self) -> _DriverLoadBalancer:
        return self._lb

    def driver(self, name: str) -> LlmDriver:
        """Resolve a driver instance by name. Cached after first creation."""
        if name not in self._instances:
            class_path = self.DRIVER_MAP.get(name)
            if not class_path:
                raise ValueError(f"Unknown AI driver: {name}")

            # Dynamic import — lazy-load drivers to avoid import errors for missing SDKs
            module_path, class_name = class_path.rsplit(".", 1)
            import importlib
            module = importlib.import_module(module_path)
            cls = getattr(module, class_name)
            self._instances[name] = cls()

        return self._instances[name]

    def driver_chain(self) -> list[str]:
        """Get the ordered list of drivers to try (primary + fallback)."""
        s = get_settings()
        chain = [s.ai_driver]
        if s.ai_fallback_driver and s.ai_fallback_driver != s.ai_driver:
            chain.append(s.ai_fallback_driver)
        return chain

    # Keys that are routing/meta concerns and should never leak to drivers
    _STRIP_KEYS = {"messages", "driver_override", "model_override", "driver", "model_name", "fallback_chain"}

    def _clean_options(self, options: dict, model_override: str | None = None) -> dict:
        """Remove routing/meta keys and inject model override."""
        cleaned = {k: v for k, v in options.items() if k not in self._STRIP_KEYS}
        if model_override:
            cleaned["model"] = model_override
        return cleaned

    async def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        driver_override: str | None = None,
        model_override: str | None = None,
        **options
    ) -> LlmResponse:
        """Run completion through the override or fallback chain."""
        opts = self._clean_options(options, model_override)

        if driver_override:
            return await self._run_targeted(
                driver_override,
                lambda d: d.complete(system_prompt, user_prompt, **opts),
                "complete"
            )

        return await self._run_with_fallback(
            lambda d: d.complete(system_prompt, user_prompt, **opts),
            "complete",
        )

    async def chat(
        self,
        messages: list[dict],
        driver_override: str | None = None,
        model_override: str | None = None,
        **options
    ) -> LlmResponse:
        """Run chat through the override or fallback chain."""
        opts = self._clean_options(options, model_override)

        if driver_override:
            return await self._run_targeted(
                driver_override,
                lambda d: d.chat(messages, **opts),
                "chat"
            )

        return await self._run_with_fallback(
            lambda d: d.chat(messages, **opts),
            "chat",
        )

    async def stream(
        self,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        messages: list[dict] | None = None,
        driver_override: str | None = None,
        model_override: str | None = None,
        fallback_chain: list[str] | None = None,
        **options
    ) -> AsyncGenerator[str, None]:
        """Stream through the override or fallback chain.

        When `fallback_chain` is provided (from Brain's YAML driver_chains),
        cascades through all providers in the chain. The LoadBalancer
        reorders the chain to skip overloaded drivers (smart overflow).
        """
        opts = self._clean_options(options, model_override)

        # Support both (system, user) and (messages) formats
        if not messages:
            messages = [
                {"role": "system", "content": system_prompt or "You are a helpful assistant."},
                {"role": "user", "content": user_prompt or ""}
            ]

        # ── Resilient Chain Streaming (with LoadBalancer) ──
        if fallback_chain and len(fallback_chain) > 1:
            # Reorder chain based on capacity + health
            smart_chain = self._lb.pick_first_available(fallback_chain, self._circuit)

            last_error = None
            for i, driver_name in enumerate(smart_chain):
                if not self._circuit.is_available(driver_name):
                    continue

                # Try to acquire a load balancer slot (non-blocking)
                acquired = await self._lb.acquire(driver_name)
                if not acquired:
                    logger.info(f"LoadBalancer: {driver_name} at capacity, skipping to next")
                    continue

                t0 = time.time()
                try:
                    driver = self.driver(driver_name)
                    stream_opts = {**opts}
                    if i == 0 and model_override:
                        stream_opts["model"] = model_override
                    elif i > 0:
                        stream_opts.pop("model", None)

                    logger.info(f"DriverManager: streaming with {driver_name} (chain pos {i+1}/{len(smart_chain)})")
                    async for chunk in driver.stream(messages, **stream_opts):
                        yield chunk
                    self._circuit.record_success(driver_name)
                    self._lb.record_request(driver_name, success=True)
                    self._lb.record_latency(driver_name, (time.time() - t0) * 1000)
                    return
                except Exception as e:
                    self._circuit.record_failure(driver_name)
                    self._lb.record_request(driver_name, success=False)
                    self._lb.record_latency(driver_name, (time.time() - t0) * 1000)
                    last_error = e
                    logger.warning(f"DriverManager: {driver_name} stream failed in chain: {e}, trying next...")
                finally:
                    self._lb.release(driver_name)

            raise last_error or RuntimeError("All drivers in chain failed")

        # ── Single Driver Override (original behavior) ──
        if driver_override:
            async for chunk in self._run_targeted_stream(driver_override, messages, **opts):
                yield chunk
            return

        # ── Default Fallback Chain (settings-based) ──
        chain = self.driver_chain()
        last_error = None

        for driver_name in chain:
            if not self._circuit.is_available(driver_name):
                logger.info(f"DriverManager: {driver_name} circuit OPEN, skipping")
                continue

            acquired = await self._lb.acquire(driver_name)
            if not acquired:
                logger.info(f"LoadBalancer: {driver_name} at capacity, skipping")
                continue

            t0 = time.time()
            try:
                driver = self.driver(driver_name)
                logger.info(f"DriverManager: streaming with {driver_name}")
                async for chunk in driver.stream(messages, **opts):
                    yield chunk
                self._circuit.record_success(driver_name)
                self._lb.record_request(driver_name, success=True)
                self._lb.record_latency(driver_name, (time.time() - t0) * 1000)
                return
            except Exception as e:
                self._circuit.record_failure(driver_name)
                self._lb.record_request(driver_name, success=False)
                last_error = e
                logger.warning(f"DriverManager: {driver_name} stream failed: {e}")
            finally:
                self._lb.release(driver_name)

        raise last_error or RuntimeError("All AI drivers failed")

    async def _run_targeted(self, driver_name: str, callback, operation: str) -> LlmResponse:
        """Run a specific driver with retry, bypasses fallback."""
        acquired = await self._lb.acquire(driver_name, timeout=5.0)
        try:
            driver = self.driver(driver_name)
            result = await _retry_transient(callback, driver)
            self._circuit.record_success(driver_name)
            self._lb.record_request(driver_name, success=True)
            return result
        except Exception as e:
            self._circuit.record_failure(driver_name)
            self._lb.record_request(driver_name, success=False)
            logger.warning(f"DriverManager: targeted {driver_name} failed: {e}")
            raise
        finally:
            if acquired:
                self._lb.release(driver_name)

    async def _run_targeted_stream(self, driver_name: str, messages: list[dict], **options) -> AsyncGenerator[str, None]:
        """Run a specific driver for streaming, bypasses fallback."""
        acquired = await self._lb.acquire(driver_name, timeout=5.0)
        try:
            driver = self.driver(driver_name)
            async for chunk in driver.stream(messages, **options):
                yield chunk
            self._circuit.record_success(driver_name)
            self._lb.record_request(driver_name, success=True)
        except Exception as e:
            self._circuit.record_failure(driver_name)
            self._lb.record_request(driver_name, success=False)
            logger.warning(f"DriverManager: targeted {driver_name} stream failed: {e}")
            raise
        finally:
            if acquired:
                self._lb.release(driver_name)

    async def _run_with_fallback(self, callback, operation: str) -> LlmResponse:
        """Execute callback across the driver fallback chain with circuit breaker + retry + load balancer."""
        chain = self.driver_chain()
        last_error = None

        for driver_name in chain:
            if not self._circuit.is_available(driver_name):
                logger.info(f"DriverManager: {driver_name} circuit OPEN, skipping for {operation}")
                continue

            acquired = await self._lb.acquire(driver_name)
            if not acquired:
                logger.info(f"LoadBalancer: {driver_name} at capacity for {operation}, skipping")
                continue

            try:
                driver = self.driver(driver_name)
                logger.info(f"DriverManager: trying {driver_name} for {operation}")

                result = await _retry_transient(callback, driver)
                self._circuit.record_success(driver_name)
                self._lb.record_request(driver_name, success=True)
                return result
            except Exception as e:
                self._circuit.record_failure(driver_name)
                self._lb.record_request(driver_name, success=False)
                last_error = e
                logger.warning(f"DriverManager: {driver_name} failed for {operation}: {e}")
            finally:
                self._lb.release(driver_name)

        logger.error(f"DriverManager: all drivers failed for {operation}", extra={"drivers_tried": chain})
        raise last_error or RuntimeError("All AI drivers failed")


# ─── Singleton ──────────────────────────────────────────────────

_manager: DriverManager | None = None
_manager_lock = threading.Lock()


def get_driver_manager() -> DriverManager:
    """Get or create the singleton DriverManager."""
    global _manager
    if _manager is None:
        with _manager_lock:
            if _manager is None:
                _manager = DriverManager()
    return _manager


