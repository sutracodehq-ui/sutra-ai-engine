"""
Driver Manager — Software Factory registry for LLM providers.

Config-driven: reads `AI_DRIVER` and `AI_FALLBACK_DRIVER` from settings,
resolves driver instances by name, and provides automatic fallback chains.

Integrates CircuitBreaker (skip dead drivers) and RetryStrategy (transient fault handling).
"""

import logging

from app.config import get_settings
from app.services.drivers.base import LlmDriver, LlmResponse
from app.services.intelligence.circuit_breaker import CircuitBreaker
from app.services.intelligence.retry_strategy import RetryStrategy

logger = logging.getLogger(__name__)


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
        "mock": "app.services.drivers.mock_driver.MockDriver",
    }

    def __init__(self):
        self._instances: dict[str, LlmDriver] = {}
        self._circuit = CircuitBreaker(threshold=3, cooldown=60)
        self._retry = RetryStrategy(max_retries=2, base_delay=1.0)

    @property
    def circuit_breaker(self) -> CircuitBreaker:
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

    async def complete(self, system_prompt: str, user_prompt: str, **options) -> LlmResponse:
        """Run completion through the fallback chain."""
        return await self._run_with_fallback(
            lambda d: d.complete(system_prompt, user_prompt, **options),
            "complete",
        )

    async def chat(self, messages: list[dict], **options) -> LlmResponse:
        """Run chat through the fallback chain."""
        return await self._run_with_fallback(
            lambda d: d.chat(messages, **options),
            "chat",
        )

    async def stream(self, messages: list[dict], **options):
        """Stream through the fallback chain (tries next driver on failure)."""
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
                async for chunk in driver.stream(messages, **options):
                    yield chunk
                self._circuit.record_success(driver_name)
                return
            except Exception as e:
                self._circuit.record_failure(driver_name)
                last_error = e
                logger.warning(f"DriverManager: {driver_name} stream failed: {e}")

        raise last_error or RuntimeError("All AI drivers failed")

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
                result = await self._retry.execute(callback, driver)
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


def get_driver_manager() -> DriverManager:
    """Get or create the singleton DriverManager."""
    global _manager
    if _manager is None:
        _manager = DriverManager()
    return _manager

