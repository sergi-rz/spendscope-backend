"""Premium entitlement validation for /enrich (SPECS §11.3, §12).

Uses the RevenueCat REST API with the server **secret** key (never the public iOS key).
A user is premium if the configured entitlement is present and not expired.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import httpx

from ..config import settings

logger = logging.getLogger("spendscope.revenuecat")


class PremiumCheckUnavailable(Exception):
    """Premium gating is required but can't be verified (no key / RevenueCat unreachable)."""


async def is_premium(user_id: str) -> bool:
    if not settings.require_premium:
        return True  # dev / staging: gate disabled

    if not settings.revenuecat_secret_key:
        raise PremiumCheckUnavailable("REVENUECAT_SECRET_KEY not configured")

    url = f"{settings.revenuecat_base_url}/subscribers/{user_id}"
    headers = {"Authorization": f"Bearer {settings.revenuecat_secret_key}"}

    try:
        async with httpx.AsyncClient(timeout=settings.revenuecat_timeout_seconds) as client:
            resp = await client.get(url, headers=headers)
    except httpx.HTTPError as exc:
        raise PremiumCheckUnavailable(f"RevenueCat request failed: {exc}") from exc

    if resp.status_code == 404:
        # Unknown subscriber == no purchases == not premium.
        return False
    if resp.status_code >= 400:
        raise PremiumCheckUnavailable(f"RevenueCat returned {resp.status_code}")

    entitlements = (resp.json().get("subscriber") or {}).get("entitlements") or {}
    ent = entitlements.get(settings.revenuecat_entitlement_id)
    if not ent:
        return False
    return _is_active(ent.get("expires_date"))


def _is_active(expires_date: str | None) -> bool:
    if expires_date is None:
        return True  # lifetime / non-expiring entitlement
    try:
        expires = datetime.fromisoformat(expires_date.replace("Z", "+00:00"))
    except ValueError:
        logger.warning("Unparseable expires_date from RevenueCat: %r", expires_date)
        return False
    if expires.tzinfo is None:
        expires = expires.replace(tzinfo=timezone.utc)
    return expires > datetime.now(timezone.utc)
