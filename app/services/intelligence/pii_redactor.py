"""
PII Redactor — privacy-first data sanitization.

Shield-AI: Masks emails, phone numbers, and potential sensitive data
before sending prompts to external LLMs.
"""

import re
import logging

logger = logging.getLogger(__name__)


class PIIRedactor:
    """Utility to detect and mask PII (Personally Identifiable Information)."""

    # Regex patterns for common PII
    PATTERNS = {
        "email": r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+',
        "phone": r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',
        "credit_card": r'\b(?:\d[ -]*?){13,16}\b',
    }

    @classmethod
    def redact(cls, text: str, placeholder: str = "[REDACTED]") -> str:
        """Redact PII from the given text."""
        if not text:
            return text

        redacted_text = text
        for label, pattern in cls.PATTERNS.items():
            redacted_text = re.sub(pattern, placeholder, redacted_text)

        return redacted_text

    @classmethod
    def contains_pii(cls, text: str) -> bool:
        """Check if the text contains any PII."""
        for pattern in cls.PATTERNS.values():
            if re.search(pattern, text):
                return True
        return False
