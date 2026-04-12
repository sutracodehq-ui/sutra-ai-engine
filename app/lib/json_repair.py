"""
JSON Repair — robust JSON extraction from LLM output.

Software Factory: Centralized helper. Never call json.loads() directly
on LLM output — always use extract_json().

Handles 3 common LLM quirks:
  1. Pure JSON                        → json.loads()
  2. Markdown fences (```json {...}```) → strip fences → json.loads()
  3. JSON embedded in prose            → extract first balanced {…} → json.loads()
"""

import json
import re
import logging
from typing import Any

logger = logging.getLogger(__name__)


def extract_json(text: str) -> dict | list | None:
    """
    Extract a JSON object/array from LLM output text.

    Tries 3 strategies in order of strictness:
      1. Direct json.loads (pure JSON)
      2. Strip markdown fences and retry
      3. Find first balanced {…} block and parse it

    Returns parsed dict/list or None on failure.
    """
    if not text or not text.strip():
        return None

    text = text.strip()

    # ─── Strategy 1: Direct parse ─────────────────────
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass

    # ─── Strategy 2: Strip markdown fences ────────────
    stripped = _strip_fences(text)
    if stripped != text:
        try:
            return json.loads(stripped)
        except (json.JSONDecodeError, ValueError):
            pass

    # ─── Strategy 3: First balanced brace block ───────
    block = _extract_first_object(text)
    if block:
        try:
            return json.loads(block)
        except (json.JSONDecodeError, ValueError):
            # Try fixing trailing commas
            fixed = _fix_trailing_commas(block)
            try:
                return json.loads(fixed)
            except (json.JSONDecodeError, ValueError):
                pass

    logger.debug(f"json_repair: could not extract JSON from: {text[:200]}...")
    return None


def _strip_fences(text: str) -> str:
    """Remove ```json ... ``` or ``` ... ``` markdown code fences."""
    m = re.search(r"```(?:json|JSON)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    return m.group(1).strip() if m else text


def _extract_first_object(text: str) -> str | None:
    """Find the first balanced { ... } block in text."""
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_str = False
    esc = False

    for i in range(start, len(text)):
        ch = text[i]
        if esc:
            esc = False
            continue
        if ch == "\\":
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]

    return None


def _fix_trailing_commas(text: str) -> str:
    """Remove trailing commas before } or ] (common LLM mistake)."""
    return re.sub(r",\s*([}\]])", r"\1", text)
