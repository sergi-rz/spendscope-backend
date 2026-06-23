"""POST /parse — universal statement parsing (SPECS §11.1). Free for all users."""

from __future__ import annotations

import asyncio
import logging
import re

from fastapi import APIRouter, HTTPException

from ..config import settings
from ..schemas import ParsedTransaction, ParseMetadata, ParseRequest, ParseResponse
from ..services import llm
from ..services.preprocess import PreprocessError, prepare
from ..services.prompts import PARSE_SYSTEM
from .common import enforce_rate_limit, timed

logger = logging.getLogger("spendscope.parse")

router = APIRouter()

# A line that carries a money amount (e.g. "1.234,56" or "1,234.56"); used to find where the
# header ends and the data rows begin when chunking a flattened statement.
_DATA_LINE = re.compile(r"[0-9][0-9.,]*[.,][0-9]{2}([^0-9]|$)")


@router.post("/parse", response_model=ParseResponse)
async def parse(req: ParseRequest) -> ParseResponse:
    enforce_rate_limit(req.user_id, "parse")

    with timed("parse") as metrics:
        try:
            modality, payload = prepare(req.input_type, req.content, req.filename)
        except PreprocessError as exc:
            metrics.status = 400
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        # An image is a single multimodal call; flattened text (incl. Excel/PDF) is chunked so a big
        # statement isn't truncated by the model's output limit (#LLMBadOutput, finish_reason=length).
        if modality == "vision":
            calls = [_run_parse("Parse this bank or card statement image into transactions.", payload)]
        else:
            chunks = _chunk_statement(payload)
            calls = [_run_parse(f"Parse this statement into transactions:\n\n{chunk}", None) for chunk in chunks]

        results = await _gather_bounded(calls, settings.parse_chunk_concurrency)

        rows: list[ParsedTransaction] = []
        provider_used: str | None = None
        any_fallback = False
        primary_error: str | None = None
        succeeded = 0
        for result in results:
            if isinstance(result, Exception):
                primary_error = primary_error or str(result)
                continue
            succeeded += 1
            provider_used = provider_used or result.provider_used
            any_fallback = any_fallback or result.is_fallback
            primary_error = primary_error or result.primary_error
            rows.extend(_rows(result.data))

        if succeeded == 0:
            metrics.status = 502
            metrics.primary_error = primary_error or "all parse batches failed"
            raise HTTPException(status_code=502, detail="Parsing provider unavailable")

        metrics.provider_used = provider_used
        metrics.is_fallback = any_fallback
        metrics.primary_error = primary_error
        metrics.status = 200
        return ParseResponse(transactions=rows, metadata=ParseMetadata(bank_detected=None, count=len(rows)))


async def _run_parse(user_text: str, image_uri: str | None) -> llm.LLMResult:
    return await llm.complete_json(system=PARSE_SYSTEM, user_text=user_text, image_data_uri=image_uri)


async def _gather_bounded(coros: list, limit: int) -> list:
    """Run the parse calls with bounded concurrency, preserving order; failures come back as
    exceptions so one bad batch doesn't sink the whole import."""
    semaphore = asyncio.Semaphore(max(1, limit))

    async def _guarded(coro):
        async with semaphore:
            return await coro

    return await asyncio.gather(*[_guarded(c) for c in coros], return_exceptions=True)


def _chunk_statement(text: str) -> list[str]:
    """Split a flattened statement into batches of data lines, repeating the header on each, so each
    LLM call returns a small JSON array. Returns the text unchanged when it's small enough."""
    lines = text.splitlines()
    non_empty = [line for line in lines if line.strip()]
    if len(non_empty) <= settings.parse_chunk_line_threshold:
        return [text]

    first_data = next((i for i, line in enumerate(lines) if _DATA_LINE.search(line)), 0)
    header = [line for line in lines[:first_data] if line.strip()][:4]
    data_lines = [line for line in lines[first_data:] if line.strip()]

    size = max(1, settings.parse_chunk_size)
    chunks: list[str] = []
    for start in range(0, len(data_lines), size):
        batch = header + data_lines[start : start + size]
        chunks.append("\n".join(batch))
    return chunks


def _rows(data: dict) -> list[ParsedTransaction]:
    raw = data.get("transactions")
    if not isinstance(raw, list):
        return []

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
                    source_category=_as_str(row.get("source_category")),
                )
            )
        except (TypeError, ValueError):
            # Skip malformed rows rather than failing the whole import.
            continue
    return transactions


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
