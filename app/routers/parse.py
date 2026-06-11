"""POST /parse — universal statement parsing (SPECS §11.1). Free for all users."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from ..schemas import ParsedTransaction, ParseMetadata, ParseRequest, ParseResponse
from ..services import llm
from ..services.preprocess import PreprocessError, prepare
from ..services.prompts import PARSE_SYSTEM
from .common import enforce_rate_limit, timed

logger = logging.getLogger("spendscope.parse")

router = APIRouter()


@router.post("/parse", response_model=ParseResponse)
async def parse(req: ParseRequest) -> ParseResponse:
    enforce_rate_limit(req.user_id, "parse")

    with timed("parse") as metrics:
        try:
            modality, payload = prepare(req.input_type, req.content, req.filename)
        except PreprocessError as exc:
            metrics.status = 400
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        if modality == "vision":
            user_text = "Parse this bank or card statement image into transactions."
            image_uri: str | None = payload
        else:
            user_text = f"Parse this statement into transactions:\n\n{payload}"
            image_uri = None

        try:
            result = await llm.complete_json(
                system=PARSE_SYSTEM, user_text=user_text, image_data_uri=image_uri
            )
        except llm.LLMUnavailable as exc:
            metrics.status = 502
            metrics.primary_error = metrics.primary_error or str(exc)
            raise HTTPException(status_code=502, detail="Parsing provider unavailable") from exc

        metrics.provider_used = result.provider_used
        metrics.is_fallback = result.is_fallback
        metrics.primary_error = result.primary_error

        response = _to_response(result.data)
        metrics.status = 200
        return response


def _to_response(data: dict) -> ParseResponse:
    raw = data.get("transactions")
    if not isinstance(raw, list):
        raise HTTPException(status_code=502, detail="Parser returned no transactions array")

    transactions: list[ParsedTransaction] = []
    for row in raw:
        if not isinstance(row, dict):
            continue
        try:
            transactions.append(
                ParsedTransaction(
                    date=str(row.get("date") or ""),
                    concept=str(row.get("concept") or "").strip(),
                    amount=float(row.get("amount")),
                    balance=_as_float(row.get("balance")),
                    transaction_type=_as_str(row.get("transaction_type")),
                    notes=_as_str(row.get("notes")) or "",
                )
            )
        except (TypeError, ValueError):
            # Skip malformed rows rather than failing the whole import.
            continue

    metadata = ParseMetadata(
        bank_detected=_as_str(data.get("bank_detected")),
        count=len(transactions),
    )
    return ParseResponse(transactions=transactions, metadata=metadata)


def _as_float(value) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_str(value) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
