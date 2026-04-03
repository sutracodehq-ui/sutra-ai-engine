"""
Response Filter — Normalizes raw LLM text into structured data.

Provides robust JSON extraction, markdown stripping, and schema validation.
Absorbed from the legacy Brain component to provide a clean, standalone utility.
"""

import json
import logging
import re
from typing import Any, Optional

from app.schemas.agent_result import AgentResult

logger = logging.getLogger(__name__)


class ResponseFilter:
    """Normalizes raw LLM text → AgentResult."""

    STANDARD_FIELDS = {"suggestions"}

    def filter(self, raw: str, agent_config: dict | None = None) -> AgentResult:
        """Main entry point for filtering raw response text."""
        if not raw or not raw.strip():
            return AgentResult(data={"content": ""}, suggestions=[], raw=raw or "", parsed=False)
        
        parsed, ok = self._parse_json(raw)
        
        if not ok or not isinstance(parsed, dict):
            # Fallback for conversational responses that aren't JSON
            return AgentResult(data={"content": raw}, suggestions=[], raw=raw, parsed=False)
        
        suggestions = self._extract_suggestions(parsed)
        
        if agent_config:
            self._validate_schema(parsed, agent_config)
            
        return AgentResult(data=parsed, suggestions=suggestions, raw=raw, parsed=True)

    def _parse_json(self, raw: str) -> tuple[Any, bool]:
        """Attempts multiple strategies to extract JSON from raw text."""
        text = raw.strip()
        
        # 1. Pure JSON
        try:
            return json.loads(text), True
        except json.JSONDecodeError:
            pass
            
        # 2. Markdown Code Fences
        cleaned = self._strip_fences(text)
        if cleaned != text:
            try:
                return json.loads(cleaned), True
            except json.JSONDecodeError:
                pass
                
        # 3. First Object Detection
        obj = self._extract_json_object(text)
        if obj:
            try:
                return json.loads(obj), True
            except json.JSONDecodeError:
                pass
                
        return None, False

    def _strip_fences(self, text: str) -> str:
        """Removes ```json ... ``` blocks."""
        m = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
        return m.group(1).strip() if m else text

    def _extract_json_object(self, text: str) -> str | None:
        """Finds the first balanced brace block in text."""
        start = text.find("{")
        if start == -1:
            return None
            
        depth = 0
        in_str = esc = False
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

    def _extract_suggestions(self, data: dict) -> list[str]:
        """Pops suggestions from data record if present."""
        raw = data.pop("suggestions", None)
        if raw is None:
            return []
        if isinstance(raw, list):
            return [str(s) for s in raw if s]
        if isinstance(raw, str) and raw.strip():
            return [raw.strip()]
        return []

    def _validate_schema(self, data: dict, config: dict):
        """Logs warnings if required fields are missing from the parsed JSON."""
        schema = config.get("response_schema", {})
        if isinstance(schema, dict):
            fields = schema.get("fields", [])
        elif isinstance(schema, list):
            fields = schema
        else:
            fields = []
            
        fields = [f for f in fields if f not in self.STANDARD_FIELDS]
        if fields:
            missing = [f for f in fields if f not in data]
            if missing:
                logger.warning(f"ResponseFilter: response missing fields: {missing}")


# ─── Singleton ──────────────────────────────────────────────────

_filter: ResponseFilter | None = None

def get_response_filter() -> ResponseFilter:
    global _filter
    if _filter is None:
        _filter = ResponseFilter()
    return _filter
