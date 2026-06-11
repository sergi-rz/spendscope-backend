"""FastAPI application entry point.

Mounts the three contract endpoints under the configured API prefix (default `/api/v1`, so
the live routes are `/api/v1/parse`, `/api/v1/categorize`, `/api/v1/enrich`) plus a health
probe. Run with: `uvicorn app.main:app`.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import __version__, db
from .config import settings
from .routers import categorize, enrich, health, parse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create cache/rate-limit/log tables if the DB is reachable (no-op otherwise).
    db.init_schema()
    yield


app = FastAPI(
    title="SpendScope Backend",
    version=__version__,
    description="AI proxy for statement parsing, categorization and receipt enrichment.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# Contract endpoints (SPECS §11).
app.include_router(parse.router, prefix=settings.api_prefix, tags=["parse"])
app.include_router(categorize.router, prefix=settings.api_prefix, tags=["categorize"])
app.include_router(enrich.router, prefix=settings.api_prefix, tags=["enrich"])

# Health is also exposed under the prefix for uniform routing behind Apache.
app.include_router(health.router, tags=["health"])
app.include_router(health.router, prefix=settings.api_prefix, tags=["health"])


@app.get("/")
async def root() -> dict:
    return {"service": settings.app_name, "version": __version__}
