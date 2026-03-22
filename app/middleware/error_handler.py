"""
Global Middleware — Error Handler + Response Envelope.

Every API response is wrapped in a consistent envelope:
  { "success": true/false, "data": {...}, "error": null, "meta": {...} }

All unhandled exceptions are caught and returned as structured JSON
instead of raw 500 HTML errors.

Routes simply return their payload; the middleware wraps it automatically.
"""

import json
import time
import uuid
import logging
import traceback

from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import StreamingResponse
from pydantic import ValidationError

logger = logging.getLogger(__name__)

# Paths that should NOT be wrapped (infrastructure, docs, static)
_SKIP_PATHS = frozenset({"/health", "/ready", "/docs", "/redoc", "/openapi.json"})


class ResponseEnvelopeMiddleware(BaseHTTPMiddleware):
    """
    Wraps ALL JSON responses in a unified envelope format.

    Routes just return data:
        return {"name": "John"}          → {"success": true, "data": {"name": "John"}, ...}
        return [item1, item2]            → {"success": true, "data": [item1, item2], ...}
        return ApiResponse.ok(data=...)  → already wrapped, pass through

    Errors, validation failures, and exceptions are also wrapped via exception handlers.
    """

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint):
        request_id = str(uuid.uuid4())[:8]
        start_time = time.time()

        # Attach request_id to state for downstream use
        request.state.request_id = request_id

        try:
            response = await call_next(request)
            duration_ms = round((time.time() - start_time) * 1000, 2)

            # Skip wrapping for non-JSON, SSE, static, docs, health
            content_type = response.headers.get("content-type", "")
            if "application/json" not in content_type:
                return response
            if request.url.path in _SKIP_PATHS or request.url.path.startswith("/docs/"):
                response.headers["X-Request-Id"] = request_id
                return response

            # Read the response body
            body_bytes = b""
            async for chunk in response.body_iterator:
                if isinstance(chunk, str):
                    body_bytes += chunk.encode("utf-8")
                else:
                    body_bytes += chunk

            try:
                body = json.loads(body_bytes)
            except (json.JSONDecodeError, UnicodeDecodeError):
                # Not valid JSON — return as-is
                return _rebuild_response(response, body_bytes, request_id, duration_ms)

            # Already wrapped? (from ApiResponse.ok() or exception handlers)
            if isinstance(body, dict) and "success" in body and "data" in body and "error" in body:
                # Pass through — just add headers
                return _rebuild_response(response, body_bytes, request_id, duration_ms)

            # Wrap the raw response in the envelope
            is_error = response.status_code >= 400
            wrapped = {
                "success": not is_error,
                "data": None if is_error else body,
                "error": _build_error_from_body(body, response.status_code) if is_error else None,
                "meta": {"request_id": request_id, "duration_ms": duration_ms},
            }

            return JSONResponse(
                status_code=response.status_code,
                content=wrapped,
                headers={
                    "X-Request-Id": request_id,
                    "X-Response-Time": f"{duration_ms}ms",
                },
            )

        except Exception as exc:
            # Catch-all: this should never be hit if exception_handlers work,
            # but acts as the final safety net
            duration_ms = round((time.time() - start_time) * 1000, 2)
            logger.error(f"[{request_id}] Unhandled exception: {exc}", exc_info=True)

            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "data": None,
                    "error": {
                        "code": "INTERNAL_ERROR",
                        "message": "An unexpected error occurred.",
                        "request_id": request_id,
                    },
                    "meta": {
                        "request_id": request_id,
                        "duration_ms": duration_ms,
                    },
                },
                headers={
                    "X-Request-Id": request_id,
                    "X-Response-Time": f"{duration_ms}ms",
                },
            )


def _rebuild_response(original, body_bytes: bytes, request_id: str, duration_ms: float):
    """Rebuild a response from raw bytes, preserving status and adding headers."""
    return JSONResponse(
        status_code=original.status_code,
        content=json.loads(body_bytes),
        headers={
            "X-Request-Id": request_id,
            "X-Response-Time": f"{duration_ms}ms",
        },
    )


def _build_error_from_body(body, status_code: int) -> dict | None:
    """Extract error info from a raw error response body."""
    if isinstance(body, dict):
        return {
            "code": body.get("code", _status_to_code(status_code)),
            "message": body.get("detail", body.get("message", "Request failed")),
        }
    return {"code": _status_to_code(status_code), "message": str(body)}


# ─── Exception Handlers (registered on the app) ─────────────────

async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle FastAPI HTTPExceptions (404, 401, 403, 422, etc.)."""
    request_id = getattr(request.state, "request_id", "unknown")

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "data": None,
            "error": {
                "code": _status_to_code(exc.status_code),
                "message": exc.detail,
                "request_id": request_id,
            },
            "meta": {"request_id": request_id},
        },
    )


async def validation_exception_handler(request: Request, exc: ValidationError):
    """Handle Pydantic validation errors (malformed request body)."""
    request_id = getattr(request.state, "request_id", "unknown")

    errors = []
    for error in exc.errors():
        errors.append({
            "field": " → ".join(str(loc) for loc in error.get("loc", [])),
            "message": error.get("msg", "Invalid value"),
            "type": error.get("type", "value_error"),
        })

    return JSONResponse(
        status_code=422,
        content={
            "success": False,
            "data": None,
            "error": {
                "code": "VALIDATION_ERROR",
                "message": "Request validation failed.",
                "details": errors,
                "request_id": request_id,
            },
            "meta": {"request_id": request_id},
        },
    )


async def fastapi_validation_handler(request: Request, exc):
    """Handle FastAPI's RequestValidationError."""
    request_id = getattr(request.state, "request_id", "unknown")

    errors = []
    for error in exc.errors():
        errors.append({
            "field": " → ".join(str(loc) for loc in error.get("loc", [])),
            "message": error.get("msg", "Invalid value"),
            "type": error.get("type", "value_error"),
        })

    return JSONResponse(
        status_code=422,
        content={
            "success": False,
            "data": None,
            "error": {
                "code": "VALIDATION_ERROR",
                "message": "Request validation failed.",
                "details": errors,
                "request_id": request_id,
            },
            "meta": {"request_id": request_id},
        },
    )


async def generic_exception_handler(request: Request, exc: Exception):
    """Last-resort handler for completely unexpected exceptions."""
    request_id = getattr(request.state, "request_id", "unknown")
    logger.error(f"[{request_id}] Unhandled: {type(exc).__name__}: {exc}", exc_info=True)

    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "data": None,
            "error": {
                "code": "INTERNAL_ERROR",
                "message": "An unexpected error occurred. Please try again.",
                "request_id": request_id,
            },
            "meta": {"request_id": request_id},
        },
    )


# ─── Helpers ─────────────────────────────────────────────────────

def _status_to_code(status: int) -> str:
    """Map HTTP status codes to human-readable error codes."""
    return {
        400: "BAD_REQUEST",
        401: "UNAUTHORIZED",
        403: "FORBIDDEN",
        404: "NOT_FOUND",
        405: "METHOD_NOT_ALLOWED",
        409: "CONFLICT",
        422: "VALIDATION_ERROR",
        429: "RATE_LIMITED",
        500: "INTERNAL_ERROR",
        502: "BAD_GATEWAY",
        503: "SERVICE_UNAVAILABLE",
    }.get(status, f"HTTP_{status}")
