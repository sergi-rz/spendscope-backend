"""MySQL access for cache, rate limiting and operational logging.

The database holds NO user data — only a categorization cache (concept → category),
per-user rate-limit counters, and operational request metrics (SPECS §11.4, §11.5).

Everything here degrades gracefully: if `database_url` is empty or the DB is unreachable,
helpers fail open (cache misses, rate limit allows, logs go to stdout) so the API keeps
serving. The LLM proxy is the product; the DB is an optimization.
"""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from .config import settings

logger = logging.getLogger("spendscope.db")

_engine: Engine | None = None
_engine_initialized = False


def get_engine() -> Engine | None:
    """Lazily build a pooled engine. Returns None when the DB is disabled or unavailable."""
    global _engine, _engine_initialized
    if _engine_initialized:
        return _engine
    _engine_initialized = True

    if not settings.database_url:
        logger.info("database_url not set — running without MySQL (cache/rate-limit/log disabled)")
        _engine = None
        return None

    try:
        _engine = create_engine(
            settings.database_url,
            pool_pre_ping=True,
            pool_recycle=1800,
            pool_size=5,
            max_overflow=5,
            future=True,
        )
        # Validate the connection eagerly so a bad URL surfaces at startup, not mid-request.
        with _engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("MySQL connected")
    except Exception as exc:  # noqa: BLE001 — any driver/connection error means "no DB"
        logger.warning("MySQL unavailable, continuing without it: %s", exc)
        _engine = None
    return _engine


def execute(stmt: str, params: dict[str, Any] | None = None) -> None:
    """Fire-and-forget write. Swallows errors (logging/cache writes must never break a request)."""
    engine = get_engine()
    if engine is None:
        return
    try:
        with engine.begin() as conn:
            conn.execute(text(stmt), params or {})
    except Exception as exc:  # noqa: BLE001
        logger.warning("DB write failed (ignored): %s", exc)


def fetch_one(stmt: str, params: dict[str, Any] | None = None) -> dict[str, Any] | None:
    engine = get_engine()
    if engine is None:
        return None
    try:
        with engine.connect() as conn:
            row = conn.execute(text(stmt), params or {}).mappings().first()
            return dict(row) if row else None
    except Exception as exc:  # noqa: BLE001
        logger.warning("DB read failed (ignored): %s", exc)
        return None


SCHEMA_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS categorize_cache (
        concept_key   VARCHAR(255) NOT NULL,
        category      VARCHAR(255) NOT NULL,
        confidence    FLOAT NULL,
        updated_at    TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                      ON UPDATE CURRENT_TIMESTAMP,
        PRIMARY KEY (concept_key)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """,
    """
    CREATE TABLE IF NOT EXISTS rate_limit (
        user_id       VARCHAR(255) NOT NULL,
        endpoint      VARCHAR(64) NOT NULL,
        window_start  BIGINT NOT NULL,
        hits          INT NOT NULL DEFAULT 0,
        PRIMARY KEY (user_id, endpoint)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """,
    """
    CREATE TABLE IF NOT EXISTS request_log (
        id             BIGINT AUTO_INCREMENT PRIMARY KEY,
        ts             TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        endpoint       VARCHAR(32) NOT NULL,
        provider_used  VARCHAR(48) NULL,
        model_used     VARCHAR(64) NULL,
        is_fallback    TINYINT(1) NOT NULL DEFAULT 0,
        primary_error  VARCHAR(255) NULL,
        latency_ms     INT NULL,
        status         INT NULL,
        user_hash      VARCHAR(64) NULL,
        in_tokens      INT NULL,
        out_tokens     INT NULL,
        cost_usd       DECIMAL(12,6) NULL,
        INDEX idx_ts (ts),
        INDEX idx_endpoint (endpoint),
        INDEX idx_user (user_hash)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """,
]

# Idempotent column adds for already-deployed databases (the CREATE above only applies to a fresh
# install). MySQL has no "ADD COLUMN IF NOT EXISTS", so each runs in its own transaction and a
# duplicate-column error (1060) is swallowed — see init_schema.
MIGRATION_STATEMENTS = [
    "ALTER TABLE request_log ADD COLUMN model_used VARCHAR(64) NULL",
    "ALTER TABLE request_log ADD COLUMN user_hash VARCHAR(64) NULL",
    "ALTER TABLE request_log ADD COLUMN in_tokens INT NULL",
    "ALTER TABLE request_log ADD COLUMN out_tokens INT NULL",
    "ALTER TABLE request_log ADD COLUMN cost_usd DECIMAL(12,6) NULL",
    "ALTER TABLE request_log ADD INDEX idx_user (user_hash)",
    "ALTER TABLE request_log MODIFY COLUMN provider_used VARCHAR(48) NULL",
]


def init_schema() -> None:
    """Create tables if the DB is reachable. Safe to call repeatedly (idempotent)."""
    engine = get_engine()
    if engine is None:
        return
    try:
        with engine.begin() as conn:
            for stmt in SCHEMA_STATEMENTS:
                conn.execute(text(stmt))
        logger.info("DB schema ready")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Schema init failed (ignored): %s", exc)

    # Migrations for existing tables. Each in its own transaction so a duplicate-column/index error
    # on an already-migrated DB doesn't abort the rest.
    for stmt in MIGRATION_STATEMENTS:
        try:
            with engine.begin() as conn:
                conn.execute(text(stmt))
        except Exception:  # noqa: BLE001 — column/index already exists; expected on re-run
            pass
