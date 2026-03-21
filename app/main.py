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

    app = FastAPI(
        title="SutraAI Engine",
        description="Standalone multi-tenant AI microservice — multi-agent orchestration, self-learning, content generation",
        version="0.1.0",
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
        lifespan=lifespan,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

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
