"""
Response Filtration Engine — normalizes all LLM output into a guaranteed format.

Software Factory: This is the single point of truth for response formatting.
LLM output goes in raw → clean AgentResult comes out.
"""

import json
import logging
import re
from typing import Any

from app.schemas.agent_result import AgentResult

logger = logging.getLogger(__name__)

# ─── Singleton ───────────────────────────────────────────────────

_engine: "ResponseFilterEngine | None" = None


import json
import logging
import re
import threading
from typing import Any

from app.schemas.agent_result import AgentResult

logger = logging.getLogger(__name__)

# ─── Singleton ───────────────────────────────────────────────────

_engine: "ResponseFilterEngine | None" = None
_engine_lock = threading.Lock()


def get_response_filter() -> "ResponseFilterEngine":
    """Get or create the singleton ResponseFilterEngine."""
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                _engine = ResponseFilterEngine()
    return _engine


class ResponseFilterEngine:
    """
    Centralized engine that transforms raw LLM text into a clean,
    predictable AgentResult for the frontend.

    Pipeline:
        1. Parse   — extract JSON from raw text (handles code fences, partial JSON)
        2. Validate — check expected fields from agent's response_schema
        3. Normalize — inject defaults for missing standard fields
        4. Return   — AgentResult with .data, .suggestions, .raw, .parsed
    """

    # Standard fields that are always extracted to top-level AgentResult
    STANDARD_FIELDS = {"suggestions"}

    # ─── Public API ──────────────────────────────────────────────

    def filter(self, raw_content: str, agent_config: dict | None = None) -> AgentResult:
        """
        Filter and normalize raw LLM output.

        Args:
            raw_content: The raw text from the LLM driver.
            agent_config: The agent's YAML config dict (contains response_schema).

        Returns:
            AgentResult with guaranteed structure.
        """
        if not raw_content or not raw_content.strip():
            return AgentResult(
                data={"content": ""},
                suggestions=[],
                raw=raw_content or "",
                parsed=False,
            )

        # Step 1: Parse JSON from raw text
        parsed_data, parse_success = self._parse_json(raw_content)

        if not parse_success or not isinstance(parsed_data, dict):
            # LLM returned non-JSON — wrap as plain content
            return AgentResult(
                data={"content": raw_content},
                suggestions=[],
                raw=raw_content,
                parsed=False,
            )

        # Step 2: Extract standard fields (suggestions, etc.)
        suggestions = self._extract_suggestions(parsed_data)

        # Step 3: Validate against agent's response_schema
        if agent_config:
            parsed_data = self._validate_schema(parsed_data, agent_config)

        # Step 4: Return clean result
        return AgentResult(
            data=parsed_data,
            suggestions=suggestions,
            raw=raw_content,
            parsed=True,
        )

    # ─── Parsing ─────────────────────────────────────────────────

    def _parse_json(self, raw: str) -> tuple[Any, bool]:
        """
        Extract JSON from LLM text, handling common LLM formatting issues.

        Handles:
        - Clean JSON
        - JSON wrapped in ```json ... ``` code fences
        - JSON wrapped in ``` ... ``` code fences
        - JSON with leading/trailing text
        """
        text = raw.strip()

        # Attempt 1: Direct parse
        try:
            return json.loads(text), True
        except json.JSONDecodeError:
            pass

        # Attempt 2: Strip markdown code fences
        cleaned = self._strip_code_fences(text)
        if cleaned != text:
            try:
                return json.loads(cleaned), True
            except json.JSONDecodeError:
                pass

        # Attempt 3: Find JSON object in text (greedy brace matching)
        json_match = self._extract_json_object(text)
        if json_match:
            try:
                return json.loads(json_match), True
            except json.JSONDecodeError:
                pass

        # All parse attempts failed
        logger.debug(f"ResponseFilter: could not parse JSON from LLM output ({len(text)} chars)")
        return None, False

    def _strip_code_fences(self, text: str) -> str:
        """Remove markdown code fences (```json ... ``` or ``` ... ```)."""
        # Match ```json\n...\n``` or ```\n...\n```
        pattern = r"```(?:json)?\s*\n?(.*?)\n?\s*```"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return text

    def _extract_json_object(self, text: str) -> str | None:
        """Find the first { ... } block in text using brace counting."""
        start = text.find("{")
        if start == -1:
            return None

        depth = 0
        in_string = False
        escape_next = False

        for i in range(start, len(text)):
            ch = text[i]

            if escape_next:
                escape_next = False
                continue

            if ch == "\\":
                escape_next = True
                continue

            if ch == '"' and not escape_next:
                in_string = not in_string
                continue

            if in_string:
                continue

            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]

        return None

    # ─── Standard Field Extraction ───────────────────────────────

    def _extract_suggestions(self, data: dict) -> list[str]:
        """
        Extract and normalize the 'suggestions' field.

        Always returns a list of strings. Handles:
        - suggestions: ["a", "b"]  → as-is
        - suggestions: "single"   → ["single"]
        - suggestions: null       → []
        - suggestions missing     → []
        """
        raw_suggestions = data.pop("suggestions", None)

        if raw_suggestions is None:
            return []

        if isinstance(raw_suggestions, list):
            # Ensure all items are strings
            return [str(s) for s in raw_suggestions if s]

        if isinstance(raw_suggestions, str) and raw_suggestions.strip():
            return [raw_suggestions.strip()]

        return []

    # ─── Schema Validation ───────────────────────────────────────

    def _validate_schema(self, data: dict, agent_config: dict) -> dict:
        """
        Validate parsed data against the agent's declared response_schema.

        Doesn't reject invalid data — just logs warnings for missing fields.
        This is intentionally lenient: LLMs are unpredictable.
        """
        schema = agent_config.get("response_schema", {})

        expected_fields: list[str] = []
        if isinstance(schema, dict) and "fields" in schema:
            expected_fields = schema["fields"]
        elif isinstance(schema, list):
            expected_fields = schema

        if not expected_fields:
            return data

        # Filter out standard fields that we handle separately
        expected_fields = [f for f in expected_fields if f not in self.STANDARD_FIELDS]

        missing = [f for f in expected_fields if f not in data]
        if missing:
            logger.warning(
                f"ResponseFilter: agent response missing expected fields: {missing}"
            )

        return data
