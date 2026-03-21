"""
SutraAI Engine — FastAPI Application Entry Point.

Software Factory: the app is assembled from config-driven components.
Routes, drivers, agents all self-register at startup.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle — startup and shutdown hooks."""
    settings = get_settings()
    logger.info(f"🧠 SutraAI Engine starting (env={settings.app_env}, driver={settings.ai_driver})")

    # Pre-warm the agent hub (registers all agents)
    from app.services.agents.hub import get_agent_hub
    hub = get_agent_hub()
    logger.info(f"✅ Registered agents: {hub.available_agents()}")

    yield

    # Shutdown
    logger.info("🧠 SutraAI Engine shutting down")


def create_app() -> FastAPI:
    """Factory function — assembles the FastAPI application."""
    settings = get_settings()

    # ─── Load OpenAPI config from YAML (config-driven, not hardcoded) ───
    import yaml
    import os

    openapi_config_path = os.path.join(os.getcwd(), "config", "openapi.yaml")
    with open(openapi_config_path, "r") as f:
        openapi_cfg = yaml.safe_load(f)

    app = FastAPI(
        title=openapi_cfg.get("title", "SutraAI Engine"),
        summary=openapi_cfg.get("summary", ""),
        description=openapi_cfg.get("description", ""),
        version=openapi_cfg.get("version", "0.1.0"),
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_tags=openapi_cfg.get("tags", []),
        contact=openapi_cfg.get("contact"),
        license_info=openapi_cfg.get("license"),
        lifespan=lifespan,
    )


    # ─── Global Middleware ─────────────────────────────
    from app.middleware.error_handler import (
        ResponseEnvelopeMiddleware,
        http_exception_handler,
        validation_exception_handler,
        fastapi_validation_handler,
        generic_exception_handler,
    )
    from fastapi.exceptions import RequestValidationError
    from pydantic import ValidationError

    # Middleware (outermost = first to execute)
    app.add_middleware(ResponseEnvelopeMiddleware)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Exception handlers
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, fastapi_validation_handler)
    app.add_exception_handler(ValidationError, validation_exception_handler)
    app.add_exception_handler(Exception, generic_exception_handler)

    # ─── Mount Routes ──────────────────────────────
    from app.api.health import router as health_router
    from app.api.v1.router import router as v1_router
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import FileResponse
    import os

    app.include_router(health_router)
    app.include_router(v1_router)

    # Serve Interactive Developer Docs
    docs_path = os.path.join(os.getcwd(), "docs")
    if os.path.exists(docs_path):
        app.mount("/docs/static", StaticFiles(directory=docs_path), name="docs_static")

        @app.get("/docs/dev", include_in_schema=False)
        async def dev_docs():
            return FileResponse(os.path.join(docs_path, "index.html"))

    return app


# ─── App Instance (for uvicorn) ─────────────────────────────────

app = create_app()
