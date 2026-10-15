"""FastAPI application entry point.

Run from the ``backend/`` directory::

    uvicorn app.main:app --reload

Swagger UI is served at /docs and the OpenAPI schema at /openapi.json.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.errors import register_error_handlers
from app.api.routes import router
from app.core.config import (
    API_DESCRIPTION,
    API_TITLE,
    API_VERSION,
    cors_origins,
)
from app.core.model_registry import registry


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Load the model artifact once, before the first request."""
    registry.load()
    if registry.is_loaded:
        print(f"Model loaded (version {registry.predictor.model_version}).")
    else:
        print(f"Starting without a model: {registry.error}")
    yield


app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins(),
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

register_error_handlers(app)
app.include_router(router)
