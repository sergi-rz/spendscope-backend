"""POST /enrich — premium receipt enrichment (SPECS §11.3).

Premium-gated: the backend validates the subscription with RevenueCat. Non-premium → 403.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from ..config import settings
from ..schemas import EnrichedItem, EnrichRequest, EnrichResponse
from ..services import llm, revenuecat
from ..services.preprocess import PreprocessError, prepare
from ..services.prompts import ENRICH_SYSTEM, enrich_user_prompt
from .common import enforce_rate_limit, timed

logger = logging.getLogger("spendscope.enrich")

router = APIRouter()


@router.post("/enrich", response_model=EnrichResponse)
async def enrich(req: EnrichRequest) -> EnrichResponse:
    enforce_rate_limit(req.user_id, "enrich")

    # Premium gate first — don't spend an LLM call on a non-entitled user.
    try:
        premium = await revenuecat.is_premium(req.user_id)
    except revenuecat.PremiumCheckUnavailable as exc:
        logger.error("premium check unavailable: %s", exc)
        raise HTTPException(status_code=503, detail="Cannot verify subscription") from exc
    if not premium:
        raise HTTPException(status_code=403, detail="Premium subscription required")

    with timed("enrich") as metrics:
        try:
            # EnrichRequest carries no filename; preprocess sniffs magic bytes to route PDF/image.
            modality, payload = prepare(req.input_type, req.content)
        except PreprocessError as exc:
            metrics.status = 400
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        user_text = enrich_user_prompt(req.transaction_amount)
        if modality == "vision":
            image_uri: str | None = payload
        else:
            user_text = f"{user_text}\n\nReceipt:\n{payload}"
            image_uri = None

        try:
            result = await llm.complete_json(
                system=ENRICH_SYSTEM, user_text=user_text, image_data_uri=image_uri
            )
        except llm.LLMUnavailable as exc:
            metrics.status = 502
            metrics.primary_error = metrics.primary_error or str(exc)
            raise HTTPException(status_code=502, detail="Enrichment provider unavailable") from exc

        metrics.provider_used = result.provider_used
        metrics.is_fallback = result.is_fallback
        metrics.primary_error = result.primary_error

        response = _to_response(result.data, req.transaction_amount)
        metrics.status = 200
        return response


def _to_response(data: dict, transaction_amount: float) -> EnrichResponse:
    raw = data.get("items")
    if not isinstance(raw, list):
        raise HTTPException(status_code=502, detail="Enricher returned no items array")

    items: list[EnrichedItem] = []
    for row in raw:
        if not isinstance(row, dict):
            continue
        try:
            amount = abs(float(row.get("amount")))
        except (TypeError, ValueError):
            continue
        description = str(row.get("description") or "").strip()
        if not description:
            continue
        items.append(
            EnrichedItem(
                description=description,
                amount=round(amount, 2),
                category_suggestion=_as_str(row.get("category_suggestion")),
            )
        )

    total_parsed = _as_float(data.get("total_parsed"))
    if total_parsed is None:
        total_parsed = round(sum(i.amount for i in items), 2)

    reference = abs(transaction_amount)
    tolerance = max(reference * settings.enrich_total_tolerance, 0.01)
    matches = reference > 0 and abs(total_parsed - reference) <= tolerance

    return EnrichResponse(items=items, total_parsed=round(total_parsed, 2), matches_transaction=matches)


def _as_float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_str(value) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
