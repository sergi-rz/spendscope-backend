"""Shared helpers for the endpoint routers: rate-limit enforcement and request timing."""

from __future__ import annotations

import time
from contextlib import contextmanager

from fastapi import HTTPException

from ..services import rate_limit
from ..services.oplog import RequestMetrics, anon_user


def enforce_rate_limit(user_id: str, endpoint: str) -> None:
    try:
        rate_limit.check(user_id, endpoint)
    except rate_limit.RateLimitExceeded as exc:
        raise HTTPException(
            status_code=429,
            detail="Too many requests",
            headers={"Retry-After": str(exc.retry_after)},
        ) from exc


@contextmanager
def timed(endpoint: str, user_id: str | None = None):
    """Yield a RequestMetrics, stamping latency and recording it on exit (success or error)."""
    metrics = RequestMetrics(endpoint=endpoint, user_hash=anon_user(user_id))
    started = time.perf_counter()
    try:
        yield metrics
    finally:
        metrics.latency_ms = int((time.perf_counter() - started) * 1000)
        metrics.record()
