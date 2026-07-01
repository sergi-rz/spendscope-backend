"""Wire contracts (SPECS §11). Field names are snake_case to match exactly what the iOS app
sends and decodes (it uses convert-to/from-snake-case key strategies)."""

from __future__ import annotations

from enum import Enum
from typing import Literal

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
    # Signed fast-lane grant from /parse/plan (#speed P0). A chunk goes to the OpenAI fast lane ONLY
    # with a valid, unexpired grant for this user; without one it stays on free gemma. Prevents a
    # client from routing itself to the paid lane and bypassing the cost guardrails.
    grant: str | None = None


class ParsePlanRequest(BaseModel):
    """The app asks, before a chunked import, whether it may use the OpenAI fast lane (#speed).
    chunk_count is how many /parse calls the statement will split into."""
    user_id: str
    chunk_count: int


class ParsePlanResponse(BaseModel):
    lane: Literal["gemma", "openai"]  # "openai" = fire chunks in parallel, passing `grant` to /parse
    concurrency: int  # how many chunks the app may run at once (1 for the gemma lane)
    grant: str | None = None  # signed token to attach to each /parse call on the fast lane


class ParsedTransaction(BaseModel):
    date: str  # canonical yyyy-MM-dd
    concept: str
    amount: float
    balance: float | None = None
    transaction_type: str | None = None
    notes: str | None = ""
    # The category the file already carried (e.g. an export from another finance app like Mint/YNAB).
    # Copied verbatim from the input; null when the file has no category column (#66).
    source_category: str | None = None


class ParseMetadata(BaseModel):
    bank_detected: str | None = None
    count: int = 0


class ParseResponse(BaseModel):
    transactions: list[ParsedTransaction]
    metadata: ParseMetadata


# --- POST /extract -------------------------------------------------------------


class ExtractRequest(BaseModel):
    """Same payload as /parse, but /extract only flattens the document — it never calls the LLM."""
    user_id: str
    input_type: InputType
    content: str  # raw text, or base64 of a binary document/image
    filename: str | None = None


class ExtractResponse(BaseModel):
    """Result of flattening a binary statement (Excel/PDF) to text WITHOUT the LLM (#42). The app
    chunks `text` into small /parse calls with a progress bar. A real photo/scan can't be flattened,
    so `kind="image"` tells the app to send it straight to /parse as a single multimodal call."""
    kind: Literal["text", "image"]
    text: str | None = None


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
    # A short, stable substring that identifies the merchant/provider so the app can learn a REUSABLE
    # rule instead of the full noisy concept (#50). E.g. "amazon.es" from "www.amazon.esv1332423".
    suggested_pattern: str | None = None


# --- POST /categorize/batch ----------------------------------------------------


class CategorizeItem(BaseModel):
    concept: str
    amount: float
    transaction_type: str | None = None
    notes: str | None = None
    # The user's own category for this transaction from another app's export, when the app couldn't
    # match it to the local tree on-device. A strong hint to map or propose, not guess (#66).
    source_category: str | None = None


class CategorizeBatchRequest(BaseModel):
    """Categorize several transactions in ONE request, so a large import doesn't fan out into one
    LLM call (and one rate-limit hit) per transaction (#44). The category list is shared."""
    user_id: str
    items: list[CategorizeItem] = Field(default_factory=list)
    categories: list[str] = Field(default_factory=list)
    rejected_suggestions: list[str] = Field(default_factory=list)
    # New-category names already proposed earlier in THIS import (#dedup): the app chunks a big import
    # into several batch calls, and each call is stateless, so without this the model coins a fresh
    # variant ("Supermercado" vs "Supermercados") per chunk. We feed the running list back so the model
    # reuses an existing proposal verbatim when it fits, and the app collapses them by exact name.
    already_suggested: list[str] = Field(default_factory=list)
    language: str = "en"


class CategorizeResult(BaseModel):
    index: int  # echoes the input position so the app can map results back
    category: str | None = None
    confidence: float | None = None
    suggested_category: SuggestedCategory | None = None
    suggested_pattern: str | None = None  # reusable merchant token for rule learning (#50)


class CategorizeBatchResponse(BaseModel):
    results: list[CategorizeResult] = Field(default_factory=list)


# --- POST /enrich --------------------------------------------------------------


class EnrichRequest(BaseModel):
    user_id: str
    input_type: InputType
    content: str
    transaction_amount: float
    # Allowed category labels ("Parent / Child") so the enricher files each product under the real
    # taxonomy instead of inventing labels that don't resolve in the app (#55). Empty = free-form hint.
    categories: list[str] = Field(default_factory=list)


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
    # The receipt's own printed values (#51, subtotal also feeds the #52 OCR gate). The app anchors
    # the transaction amount/date to these
    # authoritative figures instead of summing OCR items (which may be incomplete), and shows them
    # for the user to confirm. None when not visible on the document.
    ticket_total: float | None = None   # the amount actually paid ("TOTAL A PAGAR" / "IMPORTE")
    subtotal: float | None = None       # sum before coupons/discounts ("SUBTOTAL"), if printed
    ticket_date: str | None = None      # purchase date, YYYY-MM-DD
    # Set when the document holds several independent purchases; the app then creates one
    # transaction per movement instead of one big breakdown (#35). None/empty for a single receipt.
    movements: list[EnrichMovement] | None = None


# --- Errors --------------------------------------------------------------------


class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None
