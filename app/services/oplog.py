"""Operational logging (SPECS §11.5). Records per-request metrics — never user data.

Writes to MySQL when available and always echoes a structured line to stdout so the metrics
are visible in `journalctl` even without a database.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .. import db

logger = logging.getLogger("spendscope.oplog")


@dataclass
class RequestMetrics:
    endpoint: str
    provider_used: str | None = None
    is_fallback: bool = False
    primary_error: str | None = None
    latency_ms: int | None = None
    status: int | None = None

    def record(self) -> None:
        logger.info(
            "endpoint=%s provider=%s fallback=%s primary_error=%s latency_ms=%s status=%s",
            self.endpoint,
            self.provider_used,
            self.is_fallback,
            self.primary_error,
            self.latency_ms,
            self.status,
        )
        db.execute(
            """
            INSERT INTO request_log
                (endpoint, provider_used, is_fallback, primary_error, latency_ms, status)
            VALUES
                (:endpoint, :provider_used, :is_fallback, :primary_error, :latency_ms, :status)
            """,
            {
                "endpoint": self.endpoint,
                "provider_used": self.provider_used,
                "is_fallback": 1 if self.is_fallback else 0,
                "primary_error": (self.primary_error or "")[:255] or None,
                "latency_ms": self.latency_ms,
                "status": self.status,
            },
        )
