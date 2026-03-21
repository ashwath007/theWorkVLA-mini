"""
FastAPI application entry-point for the India VLA Data Engine.

Mounts all route groups and provides health check, CORS middleware,
and OpenAPI metadata.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import sessions, pipeline, training

logger = logging.getLogger(__name__)

try:
    from .routes import ingest as ingest_router
    _HAS_INGEST = True
except ImportError as _ie:
    logger.warning("Ingest router unavailable: %s", _ie)
    _HAS_INGEST = False


# ── Application state ─────────────────────────────────────────────────────────

class AppState:
    """Container for shared app state loaded at startup."""
    data_dir: str = os.environ.get("DATA_DIR", "/data/sessions")
    model_dir: str = os.environ.get("MODEL_OUTPUT_DIR", "/models")
    db_url: str = os.environ.get("POSTGRES_URL", "")


_state = AppState()


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """Startup and shutdown tasks."""
    # ── Startup ───────────────────────────────────────────────────────────────
    import pathlib
    pathlib.Path(_state.data_dir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(_state.model_dir).mkdir(parents=True, exist_ok=True)
    logger.info("India VLA API starting. data_dir=%s  model_dir=%s",
                _state.data_dir, _state.model_dir)

    # Attach state to app instance so routes can access it
    app.state.data_dir   = _state.data_dir
    app.state.model_dir  = _state.model_dir

    yield

    # ── Shutdown ──────────────────────────────────────────────────────────────
    logger.info("India VLA API shutting down.")


# ── Application factory ───────────────────────────────────────────────────────

def create_app() -> FastAPI:
    """Build and return the FastAPI application."""
    app = FastAPI(
        title="India Egocentric VLA Data Engine",
        description=(
            "API for capturing egocentric headset data, running the VLA preprocessing "
            "pipeline, and managing fine-tuning of Vision-Language-Action models on "
            "India-specific task data."
        ),
        version="0.1.0",
        contact={
            "name":  "India VLA Team",
            "email": "vla@example.com",
        },
        license_info={
            "name": "Apache 2.0",
            "url":  "https://www.apache.org/licenses/LICENSE-2.0",
        },
        openapi_tags=[
            {"name": "health",   "description": "Health and readiness checks."},
            {"name": "sessions", "description": "Manage recorded sessions."},
            {"name": "pipeline", "description": "Trigger data processing pipeline steps."},
            {"name": "training", "description": "VLA model training and export."},
            {"name": "ingest",    "description": "Chunk ingestion from Raspberry Pi headsets."},
            {"name": "labeling",  "description": "LabelStudio integration: push frames, pull annotations, export labeled episodes."},
        ],
        lifespan=lifespan,
    )

    # ── CORS ──────────────────────────────────────────────────────────────────
    origins = os.environ.get("CORS_ORIGINS", "*").split(",")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routes ────────────────────────────────────────────────────────────────
    app.include_router(sessions.router, prefix="/sessions", tags=["sessions"])
    app.include_router(pipeline.router, prefix="/pipeline", tags=["pipeline"])
    app.include_router(training.router, prefix="/training", tags=["training"])
    if _HAS_INGEST:
        app.include_router(ingest_router.router, prefix="/api/ingest", tags=["ingest"])

    try:
        from .routes import labeling as labeling_router
        app.include_router(labeling_router.router, prefix="/labeling", tags=["labeling"])
    except ImportError as _le:
        logger.warning("Labeling router unavailable: %s", _le)

    # ── Health endpoint ───────────────────────────────────────────────────────
    @app.get("/health", tags=["health"], summary="Health check")
    async def health() -> dict:
        """Returns 200 if the API is running."""
        return {"status": "ok", "service": "india-vla-engine"}

    @app.get("/", tags=["health"], include_in_schema=False)
    async def root() -> dict:
        return {"message": "India VLA Data Engine API", "docs": "/docs"}

    return app


app = create_app()
