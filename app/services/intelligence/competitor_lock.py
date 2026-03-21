"""
Competitor Lock — Brand Protection Guardrail.

Shield-AI: Prevents the AI from mentioning or promoting competitors 
prefixed by the tenant.
"""

import logging
from typing import List

logger = logging.getLogger(__name__)


class CompetitorLock:
    """Service to identify and block competitor mentions."""

    @classmethod
    def find_infringements(cls, text: str, competitors: List[str]) -> List[str]:
        """Check if any competitor names appear in the text."""
        if not text or not competitors:
            return []

        found = []
        text_lower = text.lower()
        for comp in competitors:
            if comp.lower() in text_lower:
                found.append(comp)
        
        return found

    @classmethod
    def apply_guardrail(cls, text: str, competitors: List[str], replacement: str = "[Competitor]") -> str:
        """Replace competitor names with a placeholder or generic term."""
        import re
        
        if not text or not competitors:
            return text

        result = text
        for comp in competitors:
            # Use regex for case-insensitive replacement with word boundaries
            pattern = re.compile(re.escape(comp), re.IGNORECASE)
            result = pattern.sub(replacement, result)

        return result
