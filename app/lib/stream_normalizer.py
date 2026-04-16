"""
Streaming normalizer for markdown/JSON output.

Goals:
- Reduce broken formatting from very tiny chunks
- Keep latency low with bounded buffering
- Never block indefinitely (always flush on limits/end)
"""

from __future__ import annotations

from typing import AsyncGenerator


def detect_stream_mode(system_prompt: str | None = None) -> str:
    s = (system_prompt or "").lower()
    if "strict json" in s or "required top-level keys:" in s or "json only" in s:
        return "json"
    return "markdown"


async def normalize_stream(
    raw_stream: AsyncGenerator[str, None],
    *,
    mode: str = "markdown",
    min_emit_chars: int = 28,
    max_emit_chars: int = 240,
) -> AsyncGenerator[str, None]:
    """
    Normalize chunk boundaries with tiny buffering.

    - markdown mode: prefers newline/sentence boundaries
    - json mode: prefers commas/braces/newlines outside strings
    """
    buf = ""
    in_str = False
    esc = False

    def _json_boundary(text: str) -> bool:
        nonlocal in_str, esc
        boundary = False
        for ch in text:
            if esc:
                esc = False
                continue
            if ch == "\\":
                esc = True
                continue
            if ch == '"':
                in_str = not in_str
                continue
            if not in_str and ch in ",}\n":
                boundary = True
        return boundary

    async for chunk in raw_stream:
        if not chunk:
            continue
        chunk = chunk.replace("\r", "")
        buf += chunk

        if len(buf) >= max_emit_chars:
            yield buf
            buf = ""
            continue

        if len(buf) < min_emit_chars:
            continue

        if mode == "json":
            if _json_boundary(chunk):
                yield buf
                buf = ""
        else:
            if any(x in buf[-32:] for x in ("\n", ". ", "! ", "? ", ":\n")):
                yield buf
                buf = ""

    if buf:
        yield buf

