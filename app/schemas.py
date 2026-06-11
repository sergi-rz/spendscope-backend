"""Wire contracts (SPECS §11). Field names are snake_case to match exactly what the iOS app
sends and decodes (it uses convert-to/from-snake-case key strategies)."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class InputType(str, Enum):
    text = "text"
    image = "image"


# --- POST /parse ---------------------------------------------------------------


class ParseRequest(BaseModel):
    user_id: str
    input_type: InputType
    content: str  # raw text, or base64 of a binary document/image
    filename: str | None = None


class ParsedTransaction(BaseModel):
    date: str  # canonical yyyy-MM-dd
    concept: str
    amount: float
    balance: float | None = None
    transaction_type: str | None = None
    notes: str | None = ""


class ParseMetadata(BaseModel):
    bank_detected: str | None = None
    count: int = 0


class ParseResponse(BaseModel):
    transactions: list[ParsedTransaction]
    metadata: ParseMetadata


# --- POST /categorize ----------------------------------------------------------


class CategorizeRequest(BaseModel):
    user_id: str
    concept: str
    amount: float
    transaction_type: str | None = None
    notes: str | None = None
    categories: list[str] = Field(default_factory=list)


class CategorizeResponse(BaseModel):
    category: str | None = None
    confidence: float | None = None


# --- POST /enrich --------------------------------------------------------------


class EnrichRequest(BaseModel):
    user_id: str
    input_type: InputType
    content: str
    transaction_amount: float


class EnrichedItem(BaseModel):
    description: str
    amount: float
    category_suggestion: str | None = None


class EnrichResponse(BaseModel):
    items: list[EnrichedItem]
    total_parsed: float
    matches_transaction: bool


# --- Errors --------------------------------------------------------------------


class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None
