"""
Shared response normalization helpers.

Single source for:
- tolerant JSON extraction
- expected field specification parsing
- field presence checks (supports dotted nested paths)
"""

from __future__ import annotations

import json
from typing import Any

from app.lib.json_repair import extract_json


def parse_json_like(raw: str | None) -> Any | None:
    """Parse JSON-ish model output using tolerant extraction first, strict fallback second."""
    text = (raw or "").strip()
    if not text:
        return None
    data = extract_json(text)
    if data is not None:
        return data
    try:
        return json.loads(text)
    except Exception:
        return None


def split_expected_fields(expected_fields: Any) -> tuple[list[str], list[str]]:
    """
    Normalize expected_fields config:
    - list[str] => all required
    - {required:[...], optional:[...]} => split
    - {"field": ...} fallback => keys required
    """
    if isinstance(expected_fields, list):
        return [str(f) for f in expected_fields if f], []
    if isinstance(expected_fields, dict):
        req = expected_fields.get("required")
        opt = expected_fields.get("optional")
        if isinstance(req, list) or isinstance(opt, list):
            return (
                [str(f) for f in (req or []) if f],
                [str(f) for f in (opt or []) if f],
            )
        return [str(k) for k in expected_fields.keys() if k], []
    return [], []


def field_present(data: Any, path: str) -> bool:
    """Check if dotted path exists in dict/list payloads."""
    if not path:
        return False
    cur = data
    for part in path.split("."):
        if isinstance(cur, dict):
            if part not in cur:
                return False
            cur = cur[part]
            continue
        if isinstance(cur, list):
            if not part.isdigit():
                return False
            idx = int(part)
            if idx < 0 or idx >= len(cur):
                return False
            cur = cur[idx]
            continue
        return False
    return True

