"""POST /categorize — AI categorization fallback (SPECS §11.2). Free for all users.

The app sends category *labels* (not UUIDs) and we must return a label copied verbatim from
that list so the app can resolve it back to a Category. Results are cached by concept.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from ..schemas import CategorizeRequest, CategorizeResponse, SuggestedCategory
from ..services import cache, llm
from ..services.prompts import CATEGORIZE_SYSTEM, categorize_user_prompt
from .common import enforce_rate_limit, timed

logger = logging.getLogger("spendscope.categorize")

router = APIRouter()


@router.post("/categorize", response_model=CategorizeResponse)
async def categorize(req: CategorizeRequest) -> CategorizeResponse:
    enforce_rate_limit(req.user_id, "categorize")

    if not req.categories:
        raise HTTPException(status_code=400, detail="categories must not be empty")

    allowed = {c.strip().lower(): c for c in req.categories}

    with timed("categorize") as metrics:
        # Cache hit: only reuse a label the app still offers (renamed categories never resurface).
        cached = cache.get(req.concept, req.amount)
        if cached and cached[0].strip().lower() in allowed:
            metrics.provider_used = "cache"
            metrics.status = 200
            return CategorizeResponse(category=allowed[cached[0].strip().lower()], confidence=cached[1])

        prompt = categorize_user_prompt(
            req.concept, req.amount, req.transaction_type, req.notes, req.categories,
            req.rejected_suggestions, req.language,
        )
        try:
            result = await llm.complete_json(system=CATEGORIZE_SYSTEM, user_text=prompt)
        except llm.LLMUnavailable as exc:
            # Graceful: the app leaves the transaction uncategorized (SPECS §6.1 step 5).
            metrics.status = 502
            metrics.primary_error = metrics.primary_error or str(exc)
            raise HTTPException(status_code=502, detail="Categorization provider unavailable") from exc

        metrics.provider_used = result.provider_used
        metrics.is_fallback = result.is_fallback
        metrics.primary_error = result.primary_error

        category = _resolve_label(result.data.get("category"), allowed)
        confidence = _as_confidence(result.data.get("confidence"))
        suggestion = _parse_suggestion(
            result.data.get("suggested_category"), req.categories, req.rejected_suggestions
        )

        if category is not None:
            cache.put(req.concept, req.amount, category, confidence)

        metrics.status = 200
        return CategorizeResponse(
            category=category, confidence=confidence, suggested_category=suggestion
        )


def _resolve_label(value, allowed: dict[str, str]) -> str | None:
    """Map the model's answer back to an exact allowed label, or None if it doesn't match."""
    if not isinstance(value, str):
        return None
    return allowed.get(value.strip().lower())


def _parse_suggestion(
    value, categories: list[str], rejected: list[str]
) -> SuggestedCategory | None:
    """Validate a proposed new category: must have a name, not duplicate an existing label, and
    not be one the user already rejected. Returns None otherwise (so we never push noise)."""
    if not isinstance(value, dict):
        return None
    name = value.get("name")
    if not isinstance(name, str) or not name.strip():
        return None
    name = name.strip()
    key = name.lower()

    # Existing labels, by full path and by leaf (the part after "/"), to catch duplicates.
    existing = set()
    for label in categories:
        existing.add(label.strip().lower())
        existing.add(label.split("/")[-1].strip().lower())
    if key in existing:
        return None
    if key in {r.strip().lower() for r in rejected}:
        return None

    parent = value.get("parent")
    reason = value.get("reason")
    return SuggestedCategory(
        name=name,
        parent=parent.strip() if isinstance(parent, str) and parent.strip() else None,
        reason=reason.strip() if isinstance(reason, str) and reason.strip() else None,
    )


def _as_confidence(value) -> float | None:
    try:
        conf = float(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(1.0, conf))
