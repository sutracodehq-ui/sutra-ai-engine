"""
Driver Manager — Software Factory registry for LLM providers.

Config-driven: reads `AI_DRIVER` and `AI_FALLBACK_DRIVER` from settings,
resolves driver instances by name, and provides automatic fallback chains.

Integrates CircuitBreaker (skip dead drivers) and RetryStrategy (transient fault handling).
"""

import logging
import threading
from typing import Any, AsyncGenerator

from app.config import get_settings
from app.services.drivers.base import LlmDriver, LlmResponse
from app.services.intelligence.guardian import get_guardian

logger = logging.getLogger(__name__)


class DriverManager:
    """
    Factory registry for LLM drivers.

    Resolves drivers by name from a config-driven map.
    Provides fallback chain execution — if primary fails, tries fallback automatically.
    Integrates Guardian for resilience (CircuitBreaker + RetryStrategy).
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
        self._guardian = get_guardian()

    @property
    def circuit_breaker(self):
        return self._guardian.circuit_breaker

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
    _STRIP_KEYS = {"messages", "driver_override", "model_override", "driver", "model_name"}

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
        **options
    ) -> AsyncGenerator[str, None]:
        """Stream through the override or fallback chain."""
        opts = self._clean_options(options, model_override)

        # Support both (system, user) and (messages) formats
        if messages is None:
            messages = [
                {"role": "system", "content": system_prompt or "You are a helpful assistant."},
                {"role": "user", "content": user_prompt or ""}
            ]

        if driver_override:
            async for chunk in self._run_targeted_stream(driver_override, messages, **opts):
                yield chunk
            return

        chain = self.driver_chain()
        last_error = None

        for driver_name in chain:
            # Circuit breaker check
            if not self._guardian.circuit_breaker.is_available(driver_name):
                logger.info(f"DriverManager: {driver_name} circuit OPEN, skipping")
                continue

            try:
                driver = self.driver(driver_name)
                logger.info(f"DriverManager: streaming with {driver_name}")
                async for chunk in driver.stream(messages, **opts):
                    yield chunk
                self._guardian.circuit_breaker.record_success(driver_name)
                return
            except Exception as e:
                self._guardian.circuit_breaker.record_failure(driver_name)
                last_error = e
                logger.warning(f"DriverManager: {driver_name} stream failed: {e}")

        raise last_error or RuntimeError("All AI drivers failed")

    async def _run_targeted(self, driver_name: str, callback, operation: str) -> LlmResponse:
        """Run a specific driver with retry, bypasses fallback."""
        try:
            driver = self.driver(driver_name)
            result = await self._guardian.with_retry(callback, driver)
            self._guardian.circuit_breaker.record_success(driver_name)
            return result
        except Exception as e:
            self._guardian.circuit_breaker.record_failure(driver_name)
            logger.warning(f"DriverManager: targeted {driver_name} failed: {e}")
            raise

    async def _run_targeted_stream(self, driver_name: str, messages: list[dict], **options) -> AsyncGenerator[str, None]:
        """Run a specific driver for streaming, bypasses fallback."""
        try:
            driver = self.driver(driver_name)
            async for chunk in driver.stream(messages, **options):
                yield chunk
            self._guardian.circuit_breaker.record_success(driver_name)
        except Exception as e:
            self._guardian.circuit_breaker.record_failure(driver_name)
            logger.warning(f"DriverManager: targeted {driver_name} stream failed: {e}")
            raise

    async def _run_with_fallback(self, callback, operation: str) -> LlmResponse:
        """Execute callback across the driver fallback chain with circuit breaker + retry."""
        chain = self.driver_chain()
        last_error = None

        for driver_name in chain:
            # Circuit breaker check — skip dead drivers
            if not self._guardian.circuit_breaker.is_available(driver_name):
                logger.info(f"DriverManager: {driver_name} circuit OPEN, skipping for {operation}")
                continue

            try:
                driver = self.driver(driver_name)
                logger.info(f"DriverManager: trying {driver_name} for {operation}")

                # Retry transient errors before falling back
                result = await self._guardian.with_retry(callback, driver)
                self._guardian.circuit_breaker.record_success(driver_name)
                return result
            except Exception as e:
                self._guardian.circuit_breaker.record_failure(driver_name)
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

