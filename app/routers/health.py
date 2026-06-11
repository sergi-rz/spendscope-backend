"""Liveness / readiness probe. Cheap and unauthenticated — used by Apache/monitoring."""

from __future__ import annotations

from fastapi import APIRouter

from .. import __version__, db

router = APIRouter()


@router.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "version": __version__,
        "database": "up" if db.get_engine() is not None else "disabled",
    }
