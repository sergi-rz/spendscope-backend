"""POST /enrich — premium receipt enrichment (SPECS §11.3).

Premium-gated: the backend validates the subscription with RevenueCat. Non-premium → 403.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from ..config import settings
from ..schemas import EnrichedItem, EnrichMovement, EnrichRequest, EnrichResponse
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
                system=ENRICH_SYSTEM,
                user_text=user_text,
                image_data_uri=image_uri,
                accept=_completeness_ok,
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


def _parse_items(raw) -> list[EnrichedItem]:
    items: list[EnrichedItem] = []
    if not isinstance(raw, list):
        return items
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
    return items


def _parse_movements(raw) -> list[EnrichMovement]:
    """Parse the optional per-purchase grouping (#35). Movements with no items are dropped."""
    movements: list[EnrichMovement] = []
    if not isinstance(raw, list):
        return movements
    for row in raw:
        if not isinstance(row, dict):
            continue
        items = _parse_items(row.get("items"))
        if not items:
            continue
        amount = _as_float(row.get("amount"))
        if amount is None:
            amount = round(sum(i.amount for i in items), 2)
        movements.append(
            EnrichMovement(
                date=_as_str(row.get("date")),
                concept=_as_str(row.get("concept")),
                amount=round(abs(amount), 2),
                items=items,
            )
        )
    return movements


def _flatten_items(data: dict) -> list[EnrichedItem]:
    """The item list, drawing from `items` or, failing that, the per-movement items (#35)."""
    items = _parse_items(data.get("items"))
    if not items:
        movements = _parse_movements(data.get("movements"))
        items = [item for movement in movements for item in movement.items]
    return items


def _completeness_ok(data: dict) -> bool:
    """Quality gate for provider escalation (#52): accept a parse only if the items we extracted
    add up to the receipt's printed subtotal (or total) within tolerance. A parse that misses many
    line items — its sum falling well short of the printed figure — is soft-rejected so a stronger
    model gets a turn. With nothing printed to check against, we can't judge, so we accept."""
    if not isinstance(data, dict):
        return True  # let _to_response raise the precise error
    items = _flatten_items(data)
    if not items:
        return False  # no items at all → a stronger model may do better
    # Items sum to the pre-discount subtotal; fall back to the total only if no subtotal is printed.
    target = _as_float(data.get("subtotal")) or _as_float(data.get("ticket_total"))
    if not target or target <= 0:
        return True  # nothing authoritative to validate against
    item_sum = sum(i.amount for i in items)
    tolerance = max(abs(target) * settings.enrich_total_tolerance, 0.01)
    return abs(item_sum - abs(target)) <= tolerance


def _to_response(data: dict, transaction_amount: float) -> EnrichResponse:
    movements = _parse_movements(data.get("movements"))
    # If the model only filled the per-purchase movements, flatten them into the flat item list too,
    # so callers that just want the items (e.g. enriching an existing transaction) still work.
    items = _flatten_items(data)
    if not items:
        raise HTTPException(status_code=502, detail="Enricher returned no items array")

    total_parsed = _as_float(data.get("total_parsed"))
    if total_parsed is None:
        total_parsed = round(sum(i.amount for i in items), 2)

    ticket_total = _as_float(data.get("ticket_total"))
    subtotal = _as_float(data.get("subtotal"))
    ticket_date = _as_str(data.get("ticket_date"))
    ticket_total = round(abs(ticket_total), 2) if ticket_total is not None else None
    subtotal = round(abs(subtotal), 2) if subtotal is not None else None

    # When the document prints its own total, that's the authoritative figure to match against the
    # known statement amount; fall back to the summed items otherwise.
    reference = abs(transaction_amount)
    effective_total = ticket_total if ticket_total is not None else total_parsed
    tolerance = max(reference * settings.enrich_total_tolerance, 0.01)
    matches = reference > 0 and abs(effective_total - reference) <= tolerance

    return EnrichResponse(
        items=items,
        total_parsed=round(total_parsed, 2),
        matches_transaction=matches,
        ticket_total=ticket_total,
        subtotal=subtotal,
        ticket_date=ticket_date,
        movements=movements or None,
    )


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
