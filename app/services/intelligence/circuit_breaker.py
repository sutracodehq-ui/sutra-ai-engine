"""
Circuit Breaker — prevents calling dead LLM services.

Stability impact: stops cascading failures. If a driver fails N times
in a row, the circuit opens and all calls fail fast for a cooldown period.
After cooldown, one test call is allowed (half-open state). If it succeeds,
circuit closes. If it fails, circuit stays open.
"""

import logging
import time

logger = logging.getLogger(__name__)


class CircuitState:
    CLOSED = "closed"      # Normal — calls pass through
    OPEN = "open"          # Tripped — calls fail fast
    HALF_OPEN = "half_open"  # Testing — one call allowed


class CircuitBreaker:
    """
    Per-driver circuit breaker.

    States:
    - CLOSED: all calls pass through (normal)
    - OPEN: all calls fail fast (driver is dead)
    - HALF_OPEN: one test call allowed, if it passes → CLOSED, if fails → OPEN
    """

    def __init__(self, *, threshold: int = 3, cooldown: int = 60):
        """
        Args:
            threshold: consecutive failures before opening the circuit
            cooldown: seconds to wait before allowing a test call
        """
        self._threshold = threshold
        self._cooldown = cooldown
        self._states: dict[str, dict] = {}  # driver_name → state info

    def _get_state(self, driver: str) -> dict:
        if driver not in self._states:
            self._states[driver] = {
                "state": CircuitState.CLOSED,
                "failure_count": 0,
                "last_failure_time": 0,
                "last_success_time": 0,
            }
        return self._states[driver]

    def is_available(self, driver: str) -> bool:
        """Check if a driver is available for calls."""
        info = self._get_state(driver)

        if info["state"] == CircuitState.CLOSED:
            return True

        if info["state"] == CircuitState.OPEN:
            # Check if cooldown has elapsed → transition to half-open
            elapsed = time.time() - info["last_failure_time"]
            if elapsed >= self._cooldown:
                info["state"] = CircuitState.HALF_OPEN
                logger.info(f"CircuitBreaker: {driver} → HALF_OPEN (cooldown elapsed)")
                return True
            return False

        # HALF_OPEN — allow exactly one test call
        return True

    def record_success(self, driver: str) -> None:
        """Record a successful call — resets the circuit."""
        info = self._get_state(driver)
        was_half_open = info["state"] == CircuitState.HALF_OPEN

        info["state"] = CircuitState.CLOSED
        info["failure_count"] = 0
        info["last_success_time"] = time.time()

        if was_half_open:
            logger.info(f"CircuitBreaker: {driver} → CLOSED (test call succeeded)")

    def record_failure(self, driver: str) -> None:
        """Record a failed call — may trip the circuit."""
        info = self._get_state(driver)
        info["failure_count"] += 1
        info["last_failure_time"] = time.time()

        if info["state"] == CircuitState.HALF_OPEN:
            # Test call failed — back to OPEN
            info["state"] = CircuitState.OPEN
            logger.warning(f"CircuitBreaker: {driver} → OPEN (test call failed)")
        elif info["failure_count"] >= self._threshold:
            info["state"] = CircuitState.OPEN
            logger.warning(
                f"CircuitBreaker: {driver} → OPEN ({info['failure_count']} consecutive failures)"
            )

    def get_status(self) -> dict[str, str]:
        """Get circuit state for all tracked drivers."""
        return {driver: info["state"] for driver, info in self._states.items()}

    def reset(self, driver: str) -> None:
        """Manually reset a circuit (admin action)."""
        if driver in self._states:
            self._states[driver] = {
                "state": CircuitState.CLOSED,
                "failure_count": 0,
                "last_failure_time": 0,
                "last_success_time": time.time(),
            }
            logger.info(f"CircuitBreaker: {driver} manually reset → CLOSED")
