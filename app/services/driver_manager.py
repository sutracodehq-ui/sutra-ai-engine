"""
Driver Manager — Software Factory registry for LLM providers.

Config-driven: reads `AI_DRIVER` and `AI_FALLBACK_DRIVER` from settings,
resolves driver instances by name, and provides automatic fallback chains.

Integrates CircuitBreaker (skip dead drivers) and RetryStrategy (transient fault handling).
"""

import asyncio
import logging
import random
import threading
import time
from typing import Any, AsyncGenerator, Callable

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
    Integrates CircuitBreaker (skip dead drivers) and RetryStrategy (handle transient errors).
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

    @property
    def circuit_breaker(self) -> _CircuitBreaker:
        return self._circuit

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
        cascades through all providers in the chain instead of hard-failing
        on the first pick. This enables automatic Ollama → Cloud failover.
        """
        opts = self._clean_options(options, model_override)

        # Support both (system, user) and (messages) formats
        # CRITICAL: check `not messages` (catches None AND []) — pipeline sends [] for new conversations
        if not messages:
            messages = [
                {"role": "system", "content": system_prompt or "You are a helpful assistant."},
                {"role": "user", "content": user_prompt or ""}
            ]

        # ── Resilient Chain Streaming ──
        # When Brain provides a full driver chain, cascade through all providers.
        if fallback_chain and len(fallback_chain) > 1:
            last_error = None
            for i, driver_name in enumerate(fallback_chain):
                if not self._circuit.is_available(driver_name):
                    logger.info(f"DriverManager: {driver_name} circuit OPEN, skipping in chain")
                    continue

                try:
                    driver = self.driver(driver_name)
                    # Use model override only for the primary (first) driver
                    stream_opts = {**opts}
                    if i == 0 and model_override:
                        stream_opts["model"] = model_override
                    elif i > 0:
                        # For fallback drivers, remove model override (use their default)
                        stream_opts.pop("model", None)

                    logger.info(f"DriverManager: streaming with {driver_name} (chain pos {i+1}/{len(fallback_chain)})")
                    async for chunk in driver.stream(messages, **stream_opts):
                        yield chunk
                    self._circuit.record_success(driver_name)
                    return
                except Exception as e:
                    self._circuit.record_failure(driver_name)
                    last_error = e
                    logger.warning(f"DriverManager: {driver_name} stream failed in chain: {e}, trying next...")

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
            # Circuit breaker check
            if not self._circuit.is_available(driver_name):
                logger.info(f"DriverManager: {driver_name} circuit OPEN, skipping")
                continue

            try:
                driver = self.driver(driver_name)
                logger.info(f"DriverManager: streaming with {driver_name}")
                async for chunk in driver.stream(messages, **opts):
                    yield chunk
                self._circuit.record_success(driver_name)
                return
            except Exception as e:
                self._circuit.record_failure(driver_name)
                last_error = e
                logger.warning(f"DriverManager: {driver_name} stream failed: {e}")

        raise last_error or RuntimeError("All AI drivers failed")

    async def _run_targeted(self, driver_name: str, callback, operation: str) -> LlmResponse:
        """Run a specific driver with retry, bypasses fallback."""
        try:
            driver = self.driver(driver_name)
            result = await _retry_transient(callback, driver)
            self._circuit.record_success(driver_name)
            return result
        except Exception as e:
            self._circuit.record_failure(driver_name)
            logger.warning(f"DriverManager: targeted {driver_name} failed: {e}")
            raise

    async def _run_targeted_stream(self, driver_name: str, messages: list[dict], **options) -> AsyncGenerator[str, None]:
        """Run a specific driver for streaming, bypasses fallback."""
        try:
            driver = self.driver(driver_name)
            async for chunk in driver.stream(messages, **options):
                yield chunk
            self._circuit.record_success(driver_name)
        except Exception as e:
            self._circuit.record_failure(driver_name)
            logger.warning(f"DriverManager: targeted {driver_name} stream failed: {e}")
            raise

    async def _run_with_fallback(self, callback, operation: str) -> LlmResponse:
        """Execute callback across the driver fallback chain with circuit breaker + retry."""
        chain = self.driver_chain()
        last_error = None

        for driver_name in chain:
            # Circuit breaker check — skip dead drivers
            if not self._circuit.is_available(driver_name):
                logger.info(f"DriverManager: {driver_name} circuit OPEN, skipping for {operation}")
                continue

            try:
                driver = self.driver(driver_name)
                logger.info(f"DriverManager: trying {driver_name} for {operation}")

                # Retry transient errors before falling back
                result = await _retry_transient(callback, driver)
                self._circuit.record_success(driver_name)
                return result
            except Exception as e:
                self._circuit.record_failure(driver_name)
                last_error = e
                logger.warning(f"DriverManager: {driver_name} failed for {operation}: {e}")

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

