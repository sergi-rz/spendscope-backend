"""Operational logging (SPECS §11.5). Records per-request metrics — never user data.

Writes to MySQL when available and always echoes a structured line to stdout so the metrics
are visible in `journalctl` even without a database.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime, timezone

from .. import db

logger = logging.getLogger("spendscope.oplog")


def anon_user(user_id: str | None) -> str | None:
    """One-way hash of the client's (already anonymous) user_id, so usage can be aggregated
    per user (documents uploaded, monthly cost) without ever storing the raw client id."""
    if not user_id:
        return None
    return hashlib.sha256(user_id.encode("utf-8")).hexdigest()[:32]


def _current_ym() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m")


def fast_lane_count_this_month(user_hash: str | None) -> int:
    """How many large-import "fast lane" (OpenAI) parses this user has been granted this month.
    Fails open (0) when there's no DB, so the cost guardrail is enforced only where it can be."""
    if not user_hash:
        return 0
    row = db.fetch_one(
        "SELECT COUNT(*) AS n FROM parse_fast_grants WHERE user_hash = :u AND ym = :ym",
        {"u": user_hash, "ym": _current_ym()},
    )
    return int(row["n"]) if row and row.get("n") is not None else 0


def record_fast_lane_grant(user_hash: str | None) -> None:
    """Record that a fast-lane import was granted, so it counts against the monthly limit."""
    if not user_hash:
        return
    db.execute(
        "INSERT INTO parse_fast_grants (user_hash, ym) VALUES (:u, :ym)",
        {"u": user_hash, "ym": _current_ym()},
    )


@dataclass
class RequestMetrics:
    endpoint: str
    provider_used: str | None = None
    is_fallback: bool = False
    primary_error: str | None = None
    latency_ms: int | None = None
    status: int | None = None
    user_hash: str | None = None
    model_used: str | None = None
    in_tokens: int = 0
    out_tokens: int = 0
    cost_usd: float = 0.0

    def record(self) -> None:
        logger.info(
            "endpoint=%s provider=%s model=%s fallback=%s in=%s out=%s cost=%.6f "
            "latency_ms=%s status=%s primary_error=%s",
            self.endpoint,
            self.provider_used,
            self.model_used,
            self.is_fallback,
            self.in_tokens,
            self.out_tokens,
            self.cost_usd,
            self.latency_ms,
            self.status,
            self.primary_error,
        )
        db.execute(
            """
            INSERT INTO request_log
                (endpoint, provider_used, model_used, is_fallback, primary_error,
                 latency_ms, status, user_hash, in_tokens, out_tokens, cost_usd)
            VALUES
                (:endpoint, :provider_used, :model_used, :is_fallback, :primary_error,
                 :latency_ms, :status, :user_hash, :in_tokens, :out_tokens, :cost_usd)
            """,
            {
                "endpoint": self.endpoint,
                "provider_used": self.provider_used,
                "model_used": self.model_used,
                "is_fallback": 1 if self.is_fallback else 0,
                "primary_error": (self.primary_error or "")[:255] or None,
                "latency_ms": self.latency_ms,
                "status": self.status,
                "user_hash": self.user_hash,
                "in_tokens": self.in_tokens or None,
                "out_tokens": self.out_tokens or None,
                "cost_usd": round(self.cost_usd, 6) if self.cost_usd else None,
            },
        )
