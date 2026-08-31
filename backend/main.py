"""
IBVAP backend entrypoint.

Run with:
    uvicorn backend.main:app --reload

Phase 1 only wires up: config, logging, database init, and a health endpoint.
Camera/detection/alert/event routers are added in later phases and included
here the same way health's router is.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api import cameras, events, health
from backend.config import get_settings
from backend.models.database import init_db
from backend.utils.logger import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logger.info("Starting %s (env=%s)", settings.app_name, settings.app_env)

    init_db()
    logger.info("Database initialized at %s", settings.database_url)

    yield

    logger.info("Shutting down %s", settings.app_name)


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        description="Intelligent Border Video Analytics Platform",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Permissive for local dev; tighten before any real deployment.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router)
    app.include_router(cameras.router)
    app.include_router(events.router)

    return app


app = create_app()
