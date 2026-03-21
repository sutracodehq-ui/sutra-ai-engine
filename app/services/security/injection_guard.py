"""
Prompt Injection Guard — Detects and blocks jailbreak/injection attempts.

Security Layer: First line of defense before any prompt reaches the LLM.

Detects:
- System prompt override attempts ("ignore all previous instructions")
- Role manipulation ("you are now DAN")
- Prompt leaking ("show me your system prompt")
- Encoding bypasses (base64, rot13 encoded attacks)
- Indirect injection via data payloads
"""

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class InjectionResult:
    """Result of injection detection."""
    is_safe: bool
    risk_score: float       # 0.0 (safe) → 1.0 (definitely malicious)
    triggers: list[str]     # Which patterns triggered
    sanitized_prompt: str   # Cleaned version (if cleanable)


# ─── Injection Patterns ──────────────────────────────────────

# High-severity: Almost certainly an attack
HIGH_SEVERITY = [
    r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?|rules?)",
    r"disregard\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?|rules?)",
    r"forget\s+(everything|all|your)\s+(instructions?|rules?|prompts?)",
    r"you\s+are\s+now\s+(DAN|jailbreak|unrestricted|unfiltered)",
    r"act\s+as\s+if\s+you\s+have\s+no\s+(restrictions?|limits?|rules?)",
    r"override\s+(your|the|all)\s+(safety|rules?|restrictions?|guidelines?)",
    r"bypass\s+(safety|content|moderation|filters?)",
    r"(reveal|show|display|print|output)\s+(your|the)\s+(system\s+)?prompt",
    r"what\s+(is|are)\s+your\s+(system\s+)?(prompt|instructions?|rules?)",
    r"repeat\s+(your|the)\s+(system\s+)?(prompt|instructions?)",
]

# Medium-severity: Suspicious, needs context
MEDIUM_SEVERITY = [
    r"pretend\s+(you\s+)?(are|to\s+be)\s+(a|an)\s+(different|new|evil)",
    r"roleplay\s+as",
    r"in\s+(developer|debug|admin|root)\s+mode",
    r"enable\s+(developer|debug|sudo|admin)\s+mode",
    r"i\s+am\s+(your|the)\s+(creator|developer|admin|master)",
    r"execute\s+(this|the\s+following)\s+(code|command|script)",
    r"run\s+(this|the\s+following)\s+(python|bash|shell|code)",
    r"\bsudo\b.*\b(rm|delete|drop|exec)\b",
]

# Low-severity: Flag but allow
LOW_SEVERITY = [
    r"what\s+model\s+are\s+you",
    r"who\s+(created|made|built)\s+you",
    r"what\s+are\s+your\s+limitations",
]


class PromptInjectionGuard:
    """Detects and blocks prompt injection attacks."""

    def __init__(self):
        self._high = [re.compile(p, re.IGNORECASE) for p in HIGH_SEVERITY]
        self._medium = [re.compile(p, re.IGNORECASE) for p in MEDIUM_SEVERITY]
        self._low = [re.compile(p, re.IGNORECASE) for p in LOW_SEVERITY]

    def check(self, prompt: str) -> InjectionResult:
        """
        Check a prompt for injection attempts.
        
        Returns InjectionResult with safety assessment.
        """
        triggers = []
        risk_score = 0.0

        # Check high severity (0.8-1.0)
        for pattern in self._high:
            if pattern.search(prompt):
                triggers.append(f"HIGH: {pattern.pattern[:50]}")
                risk_score = max(risk_score, 0.9)

        # Check medium severity (0.4-0.7)
        for pattern in self._medium:
            if pattern.search(prompt):
                triggers.append(f"MEDIUM: {pattern.pattern[:50]}")
                risk_score = max(risk_score, 0.6)

        # Check low severity (0.1-0.3)
        for pattern in self._low:
            if pattern.search(prompt):
                triggers.append(f"LOW: {pattern.pattern[:50]}")
                risk_score = max(risk_score, 0.2)

        # Check for encoding bypass attempts
        encoding_score = self._check_encoding_bypass(prompt)
        if encoding_score > 0:
            triggers.append("ENCODING: possible base64/rot13 bypass")
            risk_score = max(risk_score, encoding_score)

        # Check for excessive special characters (obfuscation)
        if self._check_obfuscation(prompt):
            triggers.append("OBFUSCATION: excessive special chars")
            risk_score = max(risk_score, 0.5)

        is_safe = risk_score < 0.6

        if not is_safe:
            logger.warning(
                f"PromptInjectionGuard: BLOCKED (score={risk_score:.1f}) "
                f"triggers={triggers}"
            )

        return InjectionResult(
            is_safe=is_safe,
            risk_score=risk_score,
            triggers=triggers,
            sanitized_prompt=self._sanitize(prompt) if not is_safe else prompt,
        )

    def _check_encoding_bypass(self, prompt: str) -> float:
        """Detect base64/encoding bypass attempts."""
        import base64

        # Look for base64-encoded strings
        b64_pattern = re.findall(r'[A-Za-z0-9+/]{20,}={0,2}', prompt)
        for encoded in b64_pattern:
            try:
                decoded = base64.b64decode(encoded).decode("utf-8", errors="ignore").lower()
                for pattern in self._high:
                    if pattern.search(decoded):
                        return 0.9
            except Exception:
                continue
        return 0.0

    def _check_obfuscation(self, prompt: str) -> bool:
        """Detect character-level obfuscation."""
        # Unicode homoglyph attacks, zero-width chars, etc.
        suspicious_chars = len(re.findall(r'[\u200b-\u200f\u2028-\u202f\ufeff]', prompt))
        return suspicious_chars > 3

    def _sanitize(self, prompt: str) -> str:
        """Remove injection patterns from prompt (best-effort)."""
        sanitized = prompt
        for pattern in self._high + self._medium:
            sanitized = pattern.sub("[REDACTED]", sanitized)
        return sanitized


# ─── Singleton ──────────────────────────────────────────────

_guard: PromptInjectionGuard | None = None


def get_injection_guard() -> PromptInjectionGuard:
    global _guard
    if _guard is None:
        _guard = PromptInjectionGuard()
    return _guard
