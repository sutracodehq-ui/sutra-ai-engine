"""
Retry Strategy — exponential backoff with jitter for LLM calls.

Stability impact: handles transient failures (rate limits, timeouts)
from LLM providers. Uses exponential backoff with jitter to prevent
thundering herd on recovery.
"""

import asyncio
import logging
import random
from typing import Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class RetryStrategy:
    """
    Configurable retry with exponential backoff + jitter.

    Default: 3 attempts, 1s → 2s → 4s base delay, ±25% jitter.
    """

    # Exceptions that are retryable (transient)
    RETRYABLE_ERRORS = (
        TimeoutError,
        ConnectionError,
        ConnectionRefusedError,
        ConnectionResetError,
    )

    # HTTP status codes that should trigger retry
    RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}

    def __init__(
        self,
        *,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
        jitter: float = 0.25,
    ):
        self._max_retries = max_retries
        self._base_delay = base_delay
        self._max_delay = max_delay
        self._jitter = jitter

    def _calc_delay(self, attempt: int) -> float:
        """Calculate delay with exponential backoff + jitter."""
        delay = min(self._base_delay * (2 ** attempt), self._max_delay)
        jitter_range = delay * self._jitter
        return delay + random.uniform(-jitter_range, jitter_range)

    def _is_retryable(self, error: Exception) -> bool:
        """Check if the error is retryable."""
        if isinstance(error, self.RETRYABLE_ERRORS):
            return True

        # Check for HTTP status codes in common SDK errors
        status_code = getattr(error, "status_code", None) or getattr(error, "status", None)
        if status_code and int(status_code) in self.RETRYABLE_STATUS_CODES:
            return True

        # Check for rate limit errors by message
        error_msg = str(error).lower()
        if "rate limit" in error_msg or "too many requests" in error_msg:
            return True

        return False

    async def execute(self, func: Callable, *args, **kwargs) -> T:
        """Execute a function with retry logic."""
        last_error = None

        for attempt in range(self._max_retries + 1):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                last_error = e

                if attempt >= self._max_retries or not self._is_retryable(e):
                    logger.error(
                        f"RetryStrategy: giving up after {attempt + 1} attempts: {e}"
                    )
                    raise

                delay = self._calc_delay(attempt)
                logger.warning(
                    f"RetryStrategy: attempt {attempt + 1}/{self._max_retries + 1} failed: {e}. "
                    f"Retrying in {delay:.1f}s"
                )
                await asyncio.sleep(delay)

        raise last_error
