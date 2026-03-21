"""
Quality Gate — auto-scores AI outputs and regenerates if below threshold.

Stability impact: prevents low-quality responses from reaching the consumer.
Scores on multiple dimensions (completeness, JSON validity, length, relevance).
If score < threshold, auto-regenerates with augmented instructions.
"""

import json
import logging

from app.services.drivers.base import LlmResponse

logger = logging.getLogger(__name__)


class QualityGate:
    """
    Multi-dimensional quality scorer with auto-regeneration.

    Scoring dimensions (each 0-10):
    - completeness: does the response cover all expected fields?
    - format: is it valid JSON when expected?
    - length: is the response substantial enough?
    - coherence: basic structure and language quality checks

    Total score = weighted average, threshold-gated.
    """

    def __init__(self, *, enabled: bool = True, threshold: int = 6, max_retries: int = 1):
        self._enabled = enabled
        self._threshold = threshold
        self._max_retries = max_retries

    def score(self, response: LlmResponse, expected_fields: list[str] | None = None) -> dict:
        """
        Score a response on multiple dimensions.

        Returns: {total: float, dimensions: {name: score}, passed: bool}
        """
        content = response.content.strip()
        dimensions = {}

        # 1. Format score — is it valid JSON?
        dimensions["format"] = self._score_format(content)

        # 2. Completeness — does it have expected fields?
        dimensions["completeness"] = self._score_completeness(content, expected_fields)

        # 3. Length — is it substantial?
        dimensions["length"] = self._score_length(content)

        # 4. Coherence — basic quality signals
        dimensions["coherence"] = self._score_coherence(content)

        # Weighted average (format is most important)
        weights = {"format": 0.35, "completeness": 0.30, "length": 0.15, "coherence": 0.20}
        total = sum(dimensions[d] * weights[d] for d in dimensions) * 10

        passed = total >= self._threshold

        result = {
            "total": round(total, 1),
            "threshold": self._threshold,
            "passed": passed,
            "dimensions": {k: round(v * 10, 1) for k, v in dimensions.items()},
        }

        log_fn = logger.info if passed else logger.warning
        log_fn(f"QualityGate: score={result['total']}/{self._threshold} passed={passed}")

        return result

    def _score_format(self, content: str) -> float:
        """Score JSON validity (0.0-1.0)."""
        try:
            parsed = json.loads(content)
            if isinstance(parsed, dict) and len(parsed) > 0:
                return 1.0
            return 0.7  # Valid JSON but not a dict or empty
        except json.JSONDecodeError:
            # Not JSON — check if it's likely a conversational response
            if len(content) > 50 and not content.startswith("{"):
                return 0.6  # Conversational response, acceptable
            return 0.3  # Failed JSON when expected

    def _score_completeness(self, content: str, expected_fields: list[str] | None) -> float:
        """Score field coverage (0.0-1.0)."""
        if not expected_fields:
            return 0.8  # No expectations to check

        try:
            parsed = json.loads(content)
            if not isinstance(parsed, dict):
                return 0.5

            present = sum(1 for f in expected_fields if f in parsed)
            return present / len(expected_fields)
        except json.JSONDecodeError:
            return 0.5  # Can't check fields if not JSON

    def _score_length(self, content: str) -> float:
        """Score response length — penalize too short or empty."""
        word_count = len(content.split())
        if word_count == 0:
            return 0.0
        if word_count < 10:
            return 0.3
        if word_count < 30:
            return 0.6
        if word_count > 2000:
            return 0.8  # Slightly penalize very long responses
        return 1.0

    def _score_coherence(self, content: str) -> float:
        """Basic coherence scoring."""
        score = 1.0

        # Penalize error-like responses
        error_signals = ["I cannot", "I'm sorry", "As an AI", "I don't have", "error", "exception"]
        for signal in error_signals:
            if signal.lower() in content.lower():
                score -= 0.2

        # Penalize very repetitive content
        words = content.lower().split()
        if len(words) > 20:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.3:
                score -= 0.3  # Very repetitive

        return max(0.0, score)

    def build_retry_prompt(self, original_prompt: str, score_result: dict) -> str:
        """Build an augmented prompt for regeneration after quality failure."""
        weak_dimensions = [
            dim for dim, val in score_result["dimensions"].items() if val < 6.0
        ]

        augmentation = "\n\n--- IMPORTANT: QUALITY REQUIREMENTS ---\n"
        augmentation += f"Your previous response scored {score_result['total']}/10. "
        augmentation += "Please improve on these dimensions:\n"

        for dim in weak_dimensions:
            if dim == "format":
                augmentation += "- Respond with VALID JSON only. No markdown code fences.\n"
            elif dim == "completeness":
                augmentation += "- Include ALL required fields in your response.\n"
            elif dim == "length":
                augmentation += "- Provide more detailed, substantial content.\n"
            elif dim == "coherence":
                augmentation += "- Be more specific and actionable. Avoid generic AI disclaimers.\n"

        return original_prompt + augmentation
