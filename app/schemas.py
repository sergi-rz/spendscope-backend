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
    # Names the user already declined as new-category proposals — never propose these again (#24).
    rejected_suggestions: list[str] = Field(default_factory=list)
    # Device UI language ("en"/"es"); the model writes its proposal reason in this language (#28).
    language: str = "en"


class SuggestedCategory(BaseModel):
    """A NEW category the model thinks fits better than any existing label (#24). The app shows it
    as a "create this category?" prompt; it is never auto-applied."""
    name: str
    parent: str | None = None  # an existing top-level category it would live under, if any
    reason: str | None = None


class CategorizeResponse(BaseModel):
    category: str | None = None
    confidence: float | None = None
    suggested_category: SuggestedCategory | None = None


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


class EnrichMovement(BaseModel):
    """One independent purchase found in the document (#35). An order-history screenshot has several
    (different dates/orders); a single receipt has none (use the flat `items` instead)."""
    date: str | None = None       # YYYY-MM-DD if visible
    concept: str | None = None    # merchant or order reference
    amount: float                 # this purchase's own total (positive)
    items: list[EnrichedItem] = Field(default_factory=list)


class EnrichResponse(BaseModel):
    items: list[EnrichedItem]
    total_parsed: float
    matches_transaction: bool
    # Set when the document holds several independent purchases; the app then creates one
    # transaction per movement instead of one big breakdown (#35). None/empty for a single receipt.
    movements: list[EnrichMovement] | None = None


# --- Errors --------------------------------------------------------------------


class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None
