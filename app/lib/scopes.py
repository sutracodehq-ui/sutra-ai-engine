"""
API Key Scopes — centralized permission definitions.

Scopes control what a tenant API key can access.
Master key always has ["*"] (full access, bypasses all scope checks).

Scope format: "resource:action"
  - resource: agents, chat, voice, intelligence, content, rag, clicks, billing, url-analyzer
  - action: read, write, * (all actions)

Example scopes:
  ["*"]                         → Full access (default for all new keys)
  ["agents:*", "chat:*"]        → Can use agents and chat, nothing else
  ["agents:read"]               → Can list agents but not run them
  ["agents:read", "agents:write"] → Can list and run agents
"""

# ─── Available Scopes ───────────────────────────────────────────
# This is the canonical list. Add new scopes here when adding new features.

AVAILABLE_SCOPES = {
    # Wildcard — full access
    "*": "Full access to all resources",

    # Agent Hub
    "agents:read": "List available agents and their configs",
    "agents:write": "Run agents (execute tasks)",
    "agents:*": "Full agent access",

    # Conversational AI
    "chat:read": "Read chat history",
    "chat:write": "Send messages and create conversations",
    "chat:*": "Full chat access",

    # Intelligence Services
    "intelligence:read": "Read analysis results",
    "intelligence:write": "Run intelligence services (brand analysis, sentiment, etc.)",
    "intelligence:*": "Full intelligence access",

    # Content Generation
    "content:read": "Read generated content",
    "content:write": "Generate new content",
    "content:*": "Full content access",

    # Voice Pipeline
    "voice:read": "List voice profiles",
    "voice:write": "Upload voice, run voice pipeline",
    "voice:*": "Full voice access",

    # RAG Knowledge Base
    "rag:read": "Query knowledge base",
    "rag:write": "Upload and manage documents",
    "rag:*": "Full RAG access",

    # URL Analyzer
    "url-analyzer:read": "Read cached analysis results",
    "url-analyzer:write": "Trigger new URL analysis",
    "url-analyzer:*": "Full URL analyzer access",

    # Click Shield
    "clicks:read": "Read click logs and scores",
    "clicks:write": "Submit clicks for scoring, provide feedback",
    "clicks:*": "Full click shield access",

    # Billing & Usage
    "billing:read": "View usage and billing data",
    "billing:*": "Full billing access",

    # Conversations
    "conversations:read": "Read conversation threads",
    "conversations:write": "Create and manage conversations",
    "conversations:*": "Full conversation access",
}


# ─── Scope Checker ──────────────────────────────────────────────

def has_scope(granted_scopes: list[str] | None, required_scope: str) -> bool:
    """
    Check if a set of granted scopes satisfies the required scope.

    Rules:
      - ["*"] grants everything
      - "agents:*" grants "agents:read" and "agents:write"
      - "agents:read" only grants "agents:read"

    Args:
        granted_scopes: The scopes on the API key (e.g., ["agents:*", "chat:read"])
        required_scope: The scope required by the endpoint (e.g., "agents:write")

    Returns:
        True if access is allowed.
    """
    if not granted_scopes:
        return False

    # Wildcard = full access
    if "*" in granted_scopes:
        return True

    # Exact match
    if required_scope in granted_scopes:
        return True

    # Resource wildcard: "agents:*" covers "agents:read", "agents:write"
    resource = required_scope.split(":")[0]
    if f"{resource}:*" in granted_scopes:
        return True

    return False


# ─── Tag-to-Scope Mapping ──────────────────────────────────────
# Maps FastAPI route tags to required scopes.
# Used by the scope-checking middleware to auto-enforce permissions.

TAG_SCOPE_MAP: dict[str, tuple[str, str]] = {
    # tag            → (read_scope, write_scope)
    "agents":         ("agents:read", "agents:write"),
    "chat":           ("chat:read", "chat:write"),
    "intelligence":   ("intelligence:read", "intelligence:write"),
    "content":        ("content:read", "content:write"),
    "voice":          ("voice:read", "voice:write"),
    "rag":            ("rag:read", "rag:write"),
    "url-analyzer":   ("url-analyzer:read", "url-analyzer:write"),
    "click-shield":   ("clicks:read", "clicks:write"),
    "billing":        ("billing:read", "billing:read"),
    "conversations":  ("conversations:read", "conversations:write"),
}
