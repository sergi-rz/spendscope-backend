"""Per-user fixed-window rate limiting (SPECS §11.4).

Backed by MySQL so it works across uvicorn workers. Fails open: if the DB is unavailable
the limiter allows the request (availability over strictness — the LLM providers have their
own quotas as a backstop).
"""

from __future__ import annotations

import logging
import time

from .. import db
from ..config import settings

logger = logging.getLogger("spendscope.ratelimit")


class RateLimitExceeded(Exception):
    def __init__(self, retry_after: int):
        self.retry_after = retry_after
        super().__init__("rate limit exceeded")


def check(user_id: str, endpoint: str) -> None:
    """Raise RateLimitExceeded if `user_id` is over the limit for `endpoint`. No-op when disabled."""
    if not settings.rate_limit_enabled:
        return
    engine = db.get_engine()
    if engine is None:
        return  # fail open

    window = settings.rate_limit_window_seconds
    now = int(time.time())
    window_start = now - (now % window)

    try:
        with engine.begin() as conn:
            from sqlalchemy import text

            # Reset the counter when we roll into a new window, then increment atomically.
            conn.execute(
                text(
                    """
                    INSERT INTO rate_limit (user_id, endpoint, window_start, hits)
                    VALUES (:uid, :ep, :ws, 1)
                    ON DUPLICATE KEY UPDATE
                        hits = IF(window_start = :ws, hits + 1, 1),
                        window_start = :ws
                    """
                ),
                {"uid": user_id, "ep": endpoint, "ws": window_start},
            )
            row = conn.execute(
                text(
                    "SELECT hits FROM rate_limit WHERE user_id = :uid AND endpoint = :ep"
                ),
                {"uid": user_id, "ep": endpoint},
            ).first()
        hits = row[0] if row else 1
    except Exception as exc:  # noqa: BLE001
        logger.warning("rate limit check failed (allowing request): %s", exc)
        return

    if hits > settings.rate_limit_max_requests:
        retry_after = window - (now % window)
        raise RateLimitExceeded(retry_after=retry_after)
