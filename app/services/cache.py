"""Categorization cache (SPECS §4.2, §11.4): identical concept → skip the LLM call.

Keyed by a normalized concept so trivial spacing/case differences still hit. Fails open
(no DB → always a miss). The available category labels are part of the key implicitly: a
cached label is only reused if it's still one of the labels the app offers (checked by the
caller), so renamed/removed categories never resurface a stale value.
"""

from __future__ import annotations

import re
import time

from .. import db
from ..config import settings

_WS = re.compile(r"\s+")


def concept_key(concept: str, amount: float) -> str:
    normalized = _WS.sub(" ", concept or "").strip().lower()
    sign = "neg" if amount < 0 else "pos"
    return f"{sign}:{normalized}"


def get(concept: str, amount: float) -> tuple[str, float | None] | None:
    if not settings.cache_enabled:
        return None
    row = db.fetch_one(
        """
        SELECT category, confidence,
               UNIX_TIMESTAMP(updated_at) AS updated_ts
        FROM categorize_cache
        WHERE concept_key = :k
        """,
        {"k": concept_key(concept, amount)},
    )
    if not row:
        return None
    if settings.cache_ttl_seconds > 0:
        age = time.time() - float(row.get("updated_ts") or 0)
        if age > settings.cache_ttl_seconds:
            return None
    return row["category"], row.get("confidence")


def put(concept: str, amount: float, category: str, confidence: float | None) -> None:
    if not settings.cache_enabled or not category:
        return
    db.execute(
        """
        INSERT INTO categorize_cache (concept_key, category, confidence)
        VALUES (:k, :c, :conf)
        ON DUPLICATE KEY UPDATE category = :c, confidence = :conf
        """,
        {"k": concept_key(concept, amount), "c": category, "conf": confidence},
    )
