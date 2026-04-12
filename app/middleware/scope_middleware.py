"""
Scope Enforcement Middleware — auto-enforces access_control.yaml on every request.

How it works:
1. Request arrives → resolve the route's tags from FastAPI route metadata
2. Look up required scope for the tag + HTTP method from YAML
3. Check the request's resolved tenant._api_tier and ._api_scopes
4. If Effective ∩ Required = ∅ → 403 Forbidden

Zero changes to route files. Tags are read from FastAPI's route registry.
"""

import logging

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.routing import Match

from app.lib.access_engine import get_access_engine

logger = logging.getLogger(__name__)

# Paths that bypass scope checks (infra, docs, static assets)
_BYPASS_PATHS = frozenset({"/health", "/ready", "/docs", "/redoc", "/openapi.json"})


class ScopeMiddleware(BaseHTTPMiddleware):
    """
    Auto-enforces access_control.yaml scopes on every request.

    Reads the route's tags from FastAPI's route registry and checks
    the resolved tenant's tier + scopes against the YAML config.
    """

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint):
        path = request.url.path

        # ─── Bypass infrastructure paths ────────────────
        if path in _BYPASS_PATHS or path.startswith("/docs/"):
            return await call_next(request)

        # ─── Resolve route tags from FastAPI ────────────
        tags = self._resolve_tags(request)
        if not tags:
            # No tags = no scope enforcement (e.g., unknown route → 404 will handle it)
            return await call_next(request)

        engine = get_access_engine()

        # ─── Check if any tag is public ─────────────────
        for tag in tags:
            if engine.is_public(tag):
                return await call_next(request)

        # ─── Get auth metadata from request state ──────
        # These are set by get_current_tenant() in dependencies.py
        tier = getattr(request.state, "_api_tier", None)
        scopes = getattr(request.state, "_api_scopes", None)

        # If tier/scopes not yet resolved, let the request proceed
        # (the dependency will handle 401 if auth is missing)
        if tier is None:
            return await call_next(request)

        # ─── Check each tag's scope requirement ─────────
        method = request.method
        for tag in tags:
            allowed, reason = engine.check_access(scopes, tier, tag, method)
            if not allowed:
                request_id = getattr(request.state, "request_id", "unknown")
                logger.warning(
                    f"[{request_id}] ScopeMiddleware: DENIED "
                    f"tier={tier} tag={tag} method={method} reason={reason}"
                )
                return JSONResponse(
                    status_code=403,
                    content={
                        "success": False,
                        "data": None,
                        "error": {
                            "code": "FORBIDDEN",
                            "message": reason,
                            "request_id": request_id,
                        },
                        "meta": {"request_id": request_id},
                    },
                )

        return await call_next(request)

    @staticmethod
    def _resolve_tags(request: Request) -> list[str]:
        """
        Extract route tags from FastAPI's route registry.

        FastAPI stores tags on each APIRoute object. We match the
        current request path against all registered routes.
        """
        if not request.app.routes:
            return []

        for route in request.app.routes:
            match, _ = route.matches(request.scope)
            if match == Match.FULL:
                # APIRoute has .tags; Mount and other route types don't
                return getattr(route, "tags", []) or []

        return []
