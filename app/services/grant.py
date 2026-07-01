"""Signed fast-lane grants (#speed P0).

/parse used to trust a `large_import=true` boolean, so a client could route unlimited chunks to the
paid OpenAI lane without ever calling /parse/plan — bypassing the size and monthly-quota guardrails.
Now /parse/plan mints a short-lived HMAC grant bound to the (hashed) user, and /parse only takes the
fast lane for a chunk that carries a valid, unexpired grant for that same user.

The grant is stateless (no DB row per token): it just proves the gate authorized THIS user recently.
The monthly quota is still enforced at the gate (each grant issued is recorded and counted), so to
obtain grants a client must go through the gate, which caps them. HMAC is keyed by a stable secret
shared across workers (settings.parse_fast_secret, or derived from the DB URL as a default).
"""

from __future__ import annotations

import hashlib
import hmac
import time

from ..config import settings

TTL_SECONDS = 900  # 15 min — generous for a large import, short enough to bound token replay.


def _secret() -> bytes:
    raw = settings.parse_fast_secret or hashlib.sha256(
        (settings.database_url or settings.app_name).encode("utf-8")
    ).hexdigest()
    return raw.encode("utf-8")


def _sign(payload: str) -> str:
    return hmac.new(_secret(), payload.encode("utf-8"), hashlib.sha256).hexdigest()


def mint(user_hash: str, ttl_seconds: int = TTL_SECONDS) -> str:
    """Issue a grant token for `user_hash`, valid for ttl_seconds. Format: user_hash.exp.signature."""
    exp = int(time.time()) + ttl_seconds
    payload = f"{user_hash}.{exp}"
    return f"{payload}.{_sign(payload)}"


def verify(token: str | None, user_hash: str | None) -> bool:
    """True iff `token` is a well-formed, unexpired grant this server signed for `user_hash`."""
    if not token or not user_hash:
        return False
    parts = token.split(".")
    if len(parts) != 3:
        return False
    uh, exp_str, sig = parts
    if uh != user_hash:
        return False
    try:
        exp = int(exp_str)
    except ValueError:
        return False
    if exp < int(time.time()):
        return False
    return hmac.compare_digest(_sign(f"{uh}.{exp_str}"), sig)
