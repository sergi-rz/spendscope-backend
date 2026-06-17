"""POST /extract — flatten a binary statement (Excel/PDF) to text WITHOUT calling the LLM (#42).

The app uploads big spreadsheets/PDFs here first; we run the same preprocessing as /parse (PyMuPDF
for PDF, openpyxl for Excel) and return the extracted text. The app then chunks that text into small,
fast /parse calls with a progress bar — so a year-long statement never rides on one slow request.

Stateless and free: nothing is stored, no provider is called. A real photo/scan can't be flattened
to rows, so we report kind="image" and the app sends it straight to /parse as a single vision call.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from ..schemas import ExtractRequest, ExtractResponse
from ..services.preprocess import PreprocessError, prepare
from .common import enforce_rate_limit, timed

logger = logging.getLogger("spendscope.extract")

router = APIRouter()


@router.post("/extract", response_model=ExtractResponse)
async def extract(req: ExtractRequest) -> ExtractResponse:
    enforce_rate_limit(req.user_id, "extract")

    with timed("extract") as metrics:
        try:
            modality, payload = prepare(req.input_type, req.content, req.filename)
        except PreprocessError as exc:
            metrics.status = 400
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        metrics.status = 200
        if modality == "text":
            return ExtractResponse(kind="text", text=payload)
        # A real photo/scan: can't be flattened to rows; the app sends it straight to /parse.
        return ExtractResponse(kind="image", text=None)
