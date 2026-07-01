"""POST /categorize — AI categorization fallback (SPECS §11.2). Free for all users.

The app sends category *labels* (not UUIDs) and we must return a label copied verbatim from
that list so the app can resolve it back to a Category. Results are cached by concept.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from ..schemas import (
    CategorizeBatchRequest,
    CategorizeBatchResponse,
    CategorizeRequest,
    CategorizeResponse,
    CategorizeResult,
    SuggestedCategory,
)
from ..services import cache, llm, pricing
from ..services.prompts import (
    CATEGORIZE_BATCH_SYSTEM,
    CATEGORIZE_SYSTEM,
    categorize_batch_user_prompt,
    categorize_user_prompt,
)
from .common import enforce_rate_limit, timed

logger = logging.getLogger("spendscope.categorize")

router = APIRouter()


@router.post("/categorize", response_model=CategorizeResponse)
async def categorize(req: CategorizeRequest) -> CategorizeResponse:
    enforce_rate_limit(req.user_id, "categorize")

    if not req.categories:
        raise HTTPException(status_code=400, detail="categories must not be empty")

    allowed = {c.strip().lower(): c for c in req.categories}

    with timed("categorize", req.user_id) as metrics:
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
        metrics.model_used = result.model_used
        metrics.is_fallback = result.is_fallback
        metrics.primary_error = result.primary_error
        metrics.in_tokens = result.in_tokens
        metrics.out_tokens = result.out_tokens
        metrics.cost_usd = pricing.cost_usd(result.model_used, result.in_tokens, result.out_tokens)

        category = _resolve_label(result.data.get("category"), allowed)
        confidence = _as_confidence(result.data.get("confidence"))
        suggestion = _parse_suggestion(
            result.data.get("suggested_category"), req.categories, req.rejected_suggestions
        )
        pattern = _parse_pattern(result.data.get("suggested_pattern"), req.concept)

        if category is not None:
            cache.put(req.concept, req.amount, category, confidence)

        metrics.status = 200
        return CategorizeResponse(
            category=category, confidence=confidence, suggested_category=suggestion,
            suggested_pattern=pattern,
        )


@router.post("/categorize/batch", response_model=CategorizeBatchResponse)
async def categorize_batch(req: CategorizeBatchRequest) -> CategorizeBatchResponse:
    """Categorize many transactions in one shot (#44): per-item cache check first, then a SINGLE LLM
    call for the cache misses. Degrades gracefully — if the provider is down, cached items still
    resolve and the rest come back null so the app leaves them pending (never blocks the import)."""
    enforce_rate_limit(req.user_id, "categorize_batch")

    if not req.categories:
        raise HTTPException(status_code=400, detail="categories must not be empty")
    if not req.items:
        return CategorizeBatchResponse(results=[])

    allowed = {c.strip().lower(): c for c in req.categories}

    with timed("categorize_batch", req.user_id) as metrics:
        results: list[CategorizeResult | None] = [None] * len(req.items)
        misses: list[int] = []

        for i, item in enumerate(req.items):
            # Items carrying a source_category (an import from another app, #66) bypass the
            # concept cache: their mapping depends on that hint, not just the concept.
            cached = None if item.source_category else cache.get(item.concept, item.amount)
            if cached and cached[0].strip().lower() in allowed:
                results[i] = CategorizeResult(
                    index=i, category=allowed[cached[0].strip().lower()], confidence=cached[1]
                )
            else:
                misses.append(i)

        if misses:
            miss_items = [req.items[i] for i in misses]
            prompt = categorize_batch_user_prompt(
                miss_items, req.categories, req.rejected_suggestions, req.language,
                req.already_suggested,
            )
            try:
                result = await llm.complete_json(system=CATEGORIZE_BATCH_SYSTEM, user_text=prompt)
                metrics.provider_used = result.provider_used
                metrics.model_used = result.model_used
                metrics.is_fallback = result.is_fallback
                metrics.primary_error = result.primary_error
                metrics.in_tokens = result.in_tokens
                metrics.out_tokens = result.out_tokens
                metrics.cost_usd = pricing.cost_usd(
                    result.model_used, result.in_tokens, result.out_tokens
                )
                parsed = _parse_batch(result.data, allowed, req.categories, req.rejected_suggestions)
                for local_idx, global_idx in enumerate(misses):
                    category, confidence, suggestion, pattern = parsed.get(
                        local_idx, (None, None, None, None)
                    )
                    # Don't cache a source_category-driven mapping under the bare concept (#66): the
                    # same concept without a hint should still categorize on its own merits.
                    if category is not None and not req.items[global_idx].source_category:
                        cache.put(
                            req.items[global_idx].concept, req.items[global_idx].amount,
                            category, confidence,
                        )
                    results[global_idx] = CategorizeResult(
                        index=global_idx, category=category, confidence=confidence,
                        suggested_category=suggestion, suggested_pattern=pattern,
                    )
            except llm.LLMUnavailable as exc:
                # Graceful: cache hits already filled; leave misses null so the app retries later.
                metrics.primary_error = metrics.primary_error or str(exc)

        final = [
            r if r is not None else CategorizeResult(index=i, category=None)
            for i, r in enumerate(results)
        ]
        metrics.status = 200
        return CategorizeBatchResponse(results=final)


def _parse_batch(data, allowed: dict[str, str], categories: list[str], rejected: list[str]) -> dict:
    """Map the model's `results` array to {local_index: (category, confidence, suggestion, pattern)}."""
    out: dict[int, tuple] = {}
    raw = data.get("results") if isinstance(data, dict) else None
    if not isinstance(raw, list):
        return out
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        idx = entry.get("index")
        if not isinstance(idx, int) or isinstance(idx, bool):
            continue
        out[idx] = (
            _resolve_label(entry.get("category"), allowed),
            _as_confidence(entry.get("confidence")),
            _parse_suggestion(entry.get("suggested_category"), categories, rejected),
            _parse_pattern(entry.get("suggested_pattern"), None),
        )
    return out


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


def _parse_pattern(value, concept: str | None) -> str | None:
    """Validate the model's reusable rule pattern (#50): a short lowercase token. When we know the
    original concept, require the pattern to actually occur in it — drop hallucinated tokens."""
    if not isinstance(value, str):
        return None
    pattern = value.strip().lower()
    if len(pattern) < 2 or len(pattern) > 60:
        return None
    if concept is not None and pattern not in concept.lower():
        return None
    return pattern


def _as_confidence(value) -> float | None:
    try:
        conf = float(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(1.0, conf))
