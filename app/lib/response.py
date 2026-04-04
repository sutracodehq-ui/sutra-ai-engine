"""
Response Helpers — guaranteed output format for every API response.

Software Factory: NEVER return raw dicts from routes.
Always use ok(), fail(), or stream_sse().

HTTP Format:
  ok(data=...) → {success: true, data: {...}, code: 200, message: "...", meta: {...}}
  fail(...)    → {success: false, data: null, code: 4xx, message: "...", error: {...}, meta: {...}}

Stream (SSE) Format:
  data: {"type": "token", "content": "Hello"}
  data: {"type": "suggestions", "items": ["..."]}
  data: {"type": "done", "agent": "social"}
  data: {"type": "error", "message": "..."}
"""

import json
import logging
from typing import Any, AsyncGenerator, Callable

from fastapi.responses import JSONResponse, StreamingResponse

logger = logging.getLogger(__name__)


# ─── HTTP Response Helpers ──────────────────────────────────────


def ok(
    data: Any = None,
    message: str = "Success",
    code: int = 200,
    meta: dict | None = None,
) -> dict:
    """
    Build a successful API response.

    The ResponseEnvelopeMiddleware will wrap this into the full envelope.
    Routes can return this dict directly — middleware handles the rest.

    Usage:
        return ok(data=profile, message="Brand analyzed successfully")
        return ok(data=results, meta={"page": 1, "total": 100})
    """
    response = {
        "success": True,
        "data": data,
        "error": None,
        "code": code,
        "message": message,
    }
    if meta:
        response["meta"] = meta
    return response


def fail(
    message: str = "An error occurred",
    code: int = 400,
    error_code: str | None = None,
    details: Any = None,
) -> JSONResponse:
    """
    Build an error API response.

    Returns a JSONResponse directly so it bypasses normal route processing
    and gets handled by the middleware with the correct status code.

    Usage:
        return fail(message="Brand analysis timed out", code=504)
        return fail(message="Invalid scope", code=403, error_code="SCOPE_DENIED")
    """
    body = {
        "success": False,
        "data": None,
        "code": code,
        "message": message,
        "error": {
            "code": error_code or f"HTTP_{code}",
            "message": message,
        },
    }
    if details:
        body["error"]["details"] = details

    return JSONResponse(status_code=code, content=body)


# ─── SSE Stream Helper ─────────────────────────────────────────


def stream_sse(
    generator: AsyncGenerator[str, None],
    agent: str = "unknown",
    task_id: str | None = None,
) -> StreamingResponse:
    """
    Wrap an async token generator into a properly formatted SSE response.

    The generator should yield raw text tokens. This helper wraps each
    token into the standard SSE event format and adds done/error events.

    Usage:
        async def _gen():
            async for token in hub.run_stream("social", prompt, ...):
                yield token

        return stream_sse(_gen(), agent="social", task_id="abc123")

    SSE Events emitted:
        data: {"type": "token", "content": "Hello"}
        data: {"type": "suggestions", "items": ["..."]}
        data: {"type": "done", "agent": "social", "task_id": "abc123"}
        data: {"type": "error", "message": "..."}
    """
    async def _sse_wrapper():
        full_response = []
        try:
            async for token in generator:
                full_response.append(token)
                yield _sse_event("token", {"content": token})

            # Post-stream: extract suggestions from complete text
            complete_text = "".join(full_response)
            suggestions = _extract_suggestions(complete_text)

            if suggestions:
                yield _sse_event("suggestions", {"items": suggestions})

            done_payload = {"agent": agent}
            if task_id:
                done_payload["task_id"] = task_id
            yield _sse_event("done", done_payload)

        except Exception as e:
            logger.error(f"SSE stream error ({agent}): {e}")
            yield _sse_event("error", {"message": str(e)})
            done_payload = {"agent": agent}
            if task_id:
                done_payload["task_id"] = task_id
            yield _sse_event("done", done_payload)

    return StreamingResponse(
        _sse_wrapper(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def _sse_event(event_type: str, payload: dict) -> str:
    """Format a single SSE event line."""
    data = {"type": event_type, **payload}
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def _extract_suggestions(text: str) -> list[str]:
    """Try to extract suggestions from the complete response text."""
    try:
        from app.services.intelligence.response_filter import get_response_filter
        rf = get_response_filter()
        filtered = rf.filter(text)
        return filtered.suggestions or []
    except Exception:
        return []
