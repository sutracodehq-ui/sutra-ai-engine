"""Common schemas — shared response wrappers."""

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str = "ok"
    service: str = "sutra-ai"


class ReadyResponse(BaseModel):
    status: str = "ok"
    database: str = "connected"
    redis: str = "connected"
    chromadb: str = "unknown"


class ErrorResponse(BaseModel):
    detail: str
