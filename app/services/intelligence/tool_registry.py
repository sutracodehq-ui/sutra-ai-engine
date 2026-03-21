"""
Tool Registry — Built-in tools for AI agents (function calling).

Intelligence Upgrade 3: Gives agents the ability to call real tools
for grounded, factual responses. Qwen 2.5 and Groq/Gemini/Anthropic
all support function calling natively.

Tools are registered as simple Python functions with metadata
(name, description, parameters) that get passed to the LLM
as tool definitions.
"""

import json
import logging
import math
from datetime import datetime, timezone
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


# ─── Tool Definition ────────────────────────────────────────

class Tool:
    """A callable tool that an agent can invoke."""

    def __init__(
        self,
        name: str,
        description: str,
        parameters: dict,
        handler: Callable,
    ):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.handler = handler

    def to_schema(self) -> dict:
        """Convert to OpenAI function-calling schema format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    async def execute(self, **kwargs) -> str:
        """Execute the tool and return a string result."""
        try:
            result = self.handler(**kwargs)
            return json.dumps(result, ensure_ascii=False, default=str)
        except Exception as e:
            return json.dumps({"error": str(e)})


# ─── Built-in Tool Handlers ────────────────────────────────

def _calculator(expression: str) -> dict:
    """Evaluate a mathematical expression safely."""
    allowed = set("0123456789+-*/.()% ")
    if not all(c in allowed for c in expression):
        return {"error": "Invalid characters in expression"}
    try:
        result = eval(expression, {"__builtins__": {}}, {"math": math})
        return {"expression": expression, "result": result}
    except Exception as e:
        return {"error": str(e)}


def _get_current_datetime(timezone_offset: int = 0) -> dict:
    """Get current date and time."""
    now = datetime.now(timezone.utc)
    return {
        "utc": now.isoformat(),
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "day_of_week": now.strftime("%A"),
        "timestamp": int(now.timestamp()),
    }


def _json_validator(json_string: str) -> dict:
    """Validate and format a JSON string."""
    try:
        parsed = json.loads(json_string)
        return {
            "valid": True,
            "formatted": json.dumps(parsed, indent=2, ensure_ascii=False),
            "type": type(parsed).__name__,
            "keys": list(parsed.keys()) if isinstance(parsed, dict) else None,
        }
    except json.JSONDecodeError as e:
        return {"valid": False, "error": str(e)}


def _text_analyzer(text: str) -> dict:
    """Analyze text for word count, character count, reading time, etc."""
    words = text.split()
    sentences = text.count(".") + text.count("!") + text.count("?")
    return {
        "word_count": len(words),
        "character_count": len(text),
        "sentence_count": max(sentences, 1),
        "avg_word_length": round(sum(len(w) for w in words) / max(len(words), 1), 1),
        "reading_time_minutes": round(len(words) / 200, 1),
        "unique_words": len(set(w.lower() for w in words)),
    }


def _keyword_extractor(text: str, top_n: int = 10) -> dict:
    """Extract top keywords from text by frequency."""
    import re
    # Remove common stop words
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "need", "dare", "ought",
        "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
        "as", "into", "through", "during", "before", "after", "above", "below",
        "between", "out", "off", "over", "under", "again", "further", "then",
        "once", "here", "there", "when", "where", "why", "how", "all", "each",
        "every", "both", "few", "more", "most", "other", "some", "such", "no",
        "nor", "not", "only", "own", "same", "so", "than", "too", "very",
        "and", "but", "or", "if", "it", "its", "this", "that", "these", "those",
    }
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    filtered = [w for w in words if w not in stop_words]
    freq = {}
    for w in filtered:
        freq[w] = freq.get(w, 0) + 1
    sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return {"keywords": [{"word": w, "count": c} for w, c in sorted_words]}


# ─── Built-in Tools Registry ───────────────────────────────

BUILT_IN_TOOLS: dict[str, Tool] = {
    "calculator": Tool(
        name="calculator",
        description="Evaluate a mathematical expression. Use for any calculations, percentages, ROI, budgets, etc.",
        parameters={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The mathematical expression to evaluate, e.g. '(100 * 0.15) + 50'",
                },
            },
            "required": ["expression"],
        },
        handler=_calculator,
    ),
    "get_datetime": Tool(
        name="get_datetime",
        description="Get the current date and time. Use when the user asks about today, current date, deadlines, scheduling.",
        parameters={
            "type": "object",
            "properties": {
                "timezone_offset": {
                    "type": "integer",
                    "description": "Timezone offset from UTC in hours, default 0",
                },
            },
        },
        handler=_get_current_datetime,
    ),
    "json_validator": Tool(
        name="json_validator",
        description="Validate and format a JSON string. Use to check if JSON output is valid.",
        parameters={
            "type": "object",
            "properties": {
                "json_string": {
                    "type": "string",
                    "description": "The JSON string to validate",
                },
            },
            "required": ["json_string"],
        },
        handler=_json_validator,
    ),
    "text_analyzer": Tool(
        name="text_analyzer",
        description="Analyze text for word count, reading time, sentence count, etc. Use for content analysis.",
        parameters={
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The text to analyze",
                },
            },
            "required": ["text"],
        },
        handler=_text_analyzer,
    ),
    "keyword_extractor": Tool(
        name="keyword_extractor",
        description="Extract top keywords from text by frequency. Use for SEO, content analysis, topic extraction.",
        parameters={
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The text to extract keywords from",
                },
                "top_n": {
                    "type": "integer",
                    "description": "Number of top keywords to return, default 10",
                },
            },
            "required": ["text"],
        },
        handler=_keyword_extractor,
    ),
}

# ─── Agent → Tool Mapping (from intelligence_config.yaml) ──

def _load_agent_tools() -> dict[str, list[str]]:
    """Load agent-tool mappings from YAML config."""
    import yaml
    from pathlib import Path

    config_path = Path("intelligence_config.yaml")
    if not config_path.exists():
        return {}

    with open(config_path) as f:
        config = yaml.safe_load(f) or {}

    return config.get("agent_tools", {})


class ToolRegistry:
    """Central registry for agent tools."""

    def __init__(self):
        self._tools = dict(BUILT_IN_TOOLS)

    def get_tools_for_agent(self, agent_type: str) -> list[Tool]:
        """Get the tools available to a specific agent."""
        agent_tools = _load_agent_tools()
        tool_names = agent_tools.get(agent_type, [])
        return [self._tools[name] for name in tool_names if name in self._tools]

    def get_tool_schemas(self, agent_type: str) -> list[dict]:
        """Get tool schemas in OpenAI function-calling format."""
        return [tool.to_schema() for tool in self.get_tools_for_agent(agent_type)]

    def get_tool(self, name: str) -> Optional[Tool]:
        """Get a specific tool by name."""
        return self._tools.get(name)

    def register(self, tool: Tool) -> None:
        """Register a custom tool."""
        self._tools[tool.name] = tool


# ─── Singleton ──────────────────────────────────────────────
_registry: ToolRegistry | None = None


def get_tool_registry() -> ToolRegistry:
    global _registry
    if _registry is None:
        _registry = ToolRegistry()
    return _registry
