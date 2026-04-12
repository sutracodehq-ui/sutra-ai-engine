"""
ApiResponse — Unified response helper for all endpoints.

Usage:
    return ApiResponse.ok(data={"content": "..."})
    return ApiResponse.ok(data=result, meta={"tokens": 150})
    return ApiResponse.error("Agent not found", code="NOT_FOUND", status=404)
    return ApiResponse.paginated(items, total=100, page=1, per_page=20)
"""

from fastapi.responses import JSONResponse
from typing import Any


class ApiResponse:
    """Factory for consistent API responses."""

    @staticmethod
    def ok(data: Any = None, meta: dict | None = None, status: int = 200) -> JSONResponse:
        """Success response."""
        return JSONResponse(
            status_code=status,
            content={
                "success": True,
                "data": data,
                "error": None,
                "meta": meta or {},
            },
        )

    @staticmethod
    def created(data: Any = None, meta: dict | None = None) -> JSONResponse:
        """201 Created response."""
        return ApiResponse.ok(data=data, meta=meta, status=201)

    @staticmethod
    def error(
        message: str,
        code: str = "ERROR",
        status: int = 400,
        details: Any = None,
        request_id: str | None = None,
    ) -> JSONResponse:
        """Error response."""
        error_body = {
            "code": code,
            "message": message,
        }
        if details:
            error_body["details"] = details
        if request_id:
            error_body["request_id"] = request_id

        return JSONResponse(
            status_code=status,
            content={
                "success": False,
                "data": None,
                "error": error_body,
                "meta": {},
            },
        )

    @staticmethod
    def paginated(
        items: list,
        total: int,
        page: int,
        per_page: int,
        meta: dict | None = None,
    ) -> JSONResponse:
        """Paginated list response."""
        pagination_meta = {
            "total": total,
            "page": page,
            "per_page": per_page,
            "last_page": (total + per_page - 1) // per_page if per_page else 1,
        }
        if meta:
            pagination_meta.update(meta)

        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "data": items,
                "error": None,
                "meta": pagination_meta,
            },
        )
