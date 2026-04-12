"""
PII Redactor — Masks sensitive Indian PII before it reaches the LLM.

Security Layer: Prevents Aadhaar, PAN, phone, email, bank accounts
from being stored in LLM context or logs.

Supported PII for India:
- Aadhaar (12-digit)
- PAN (ABCDE1234F)
- Phone (+91 / 10-digit)
- Email
- Bank account numbers
- IFSC codes
- Credit/debit card numbers
- UPI IDs
- Passport numbers
"""

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RedactionResult:
    """Result of PII redaction."""
    original: str
    redacted: str
    pii_found: list[dict]    # [{type, value_masked, position}]
    pii_count: int


# ─── PII Patterns (India-specific) ──────────────────────────

PII_PATTERNS = {
    "aadhaar": {
        "pattern": re.compile(r'\b\d{4}\s?\d{4}\s?\d{4}\b'),
        "mask": "XXXX XXXX ****",
        "description": "Aadhaar Number",
    },
    "pan": {
        "pattern": re.compile(r'\b[A-Z]{5}\d{4}[A-Z]\b'),
        "mask": "XXXXX****X",
        "description": "PAN Card",
    },
    "phone_in": {
        "pattern": re.compile(r'(?:\+91[\s-]?)?[6-9]\d{9}\b'),
        "mask": "+91 XXXXX*****",
        "description": "Indian Phone Number",
    },
    "email": {
        "pattern": re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
        "mask": "****@****.***",
        "description": "Email Address",
    },
    "bank_account": {
        "pattern": re.compile(r'\b\d{9,18}\b'),
        "mask": "XXXXXXXX****",
        "description": "Bank Account Number",
        "context_required": ["account", "bank", "a/c", "acc"],  # Only match if context suggests bank
    },
    "ifsc": {
        "pattern": re.compile(r'\b[A-Z]{4}0[A-Z0-9]{6}\b'),
        "mask": "XXXX0XXXXXX",
        "description": "IFSC Code",
    },
    "credit_card": {
        "pattern": re.compile(r'\b(?:\d{4}[\s-]?){3}\d{4}\b'),
        "mask": "XXXX XXXX XXXX ****",
        "description": "Credit/Debit Card",
    },
    "upi": {
        "pattern": re.compile(r'\b[A-Za-z0-9._%+-]+@[a-z]{2,}\b'),
        "mask": "****@***",
        "description": "UPI ID",
        "context_required": ["upi", "pay", "gpay", "phonepe", "paytm"],
    },
    "passport": {
        "pattern": re.compile(r'\b[A-Z]\d{7}\b'),
        "mask": "X*******",
        "description": "Passport Number",
        "context_required": ["passport", "travel"],
    },
}


class PiiRedactor:
    """
    Redacts PII from text before it reaches the LLM.
    
    Features:
    - Context-aware matching (bank accounts only masked when "account" context present)
    - Preserves last 4 digits for user reference
    - Returns metadata about what was redacted
    """

    def __init__(self, enabled_types: list[str] | None = None):
        """
        Args:
            enabled_types: Which PII types to redact. None = all.
        """
        self._types = enabled_types or list(PII_PATTERNS.keys())

    def redact(self, text: str) -> RedactionResult:
        """
        Redact all PII from text.
        
        Returns RedactionResult with redacted text and metadata.
        """
        redacted = text
        pii_found = []
        text_lower = text.lower()

        for pii_type in self._types:
            config = PII_PATTERNS.get(pii_type)
            if not config:
                continue

            # Context-aware: only match if related keywords are present
            context_words = config.get("context_required", [])
            if context_words and not any(w in text_lower for w in context_words):
                continue

            for match in config["pattern"].finditer(redacted):
                original_value = match.group()

                # Skip very short matches (false positives)
                if len(original_value.replace(" ", "").replace("-", "")) < 6:
                    continue

                # Create masked value (preserve last 4 for reference)
                clean = original_value.replace(" ", "").replace("-", "")
                masked = "X" * (len(clean) - 4) + clean[-4:]

                pii_found.append({
                    "type": pii_type,
                    "description": config["description"],
                    "masked_value": masked,
                    "position": match.start(),
                })

                redacted = redacted.replace(original_value, f"[{config['description']}: {masked}]", 1)

        if pii_found:
            logger.info(f"PiiRedactor: redacted {len(pii_found)} PII items: {[p['type'] for p in pii_found]}")

        return RedactionResult(
            original=text,
            redacted=redacted,
            pii_found=pii_found,
            pii_count=len(pii_found),
        )

    def has_pii(self, text: str) -> bool:
        """Quick check — does this text contain PII?"""
        result = self.redact(text)
        return result.pii_count > 0


# ─── Singleton ──────────────────────────────────────────────

_redactor: PiiRedactor | None = None


def get_pii_redactor() -> PiiRedactor:
    global _redactor
    if _redactor is None:
        _redactor = PiiRedactor()
    return _redactor
