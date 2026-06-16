"""LLM prompts. Kept server-side by design — the app never builds prompts (SPECS §11.4).

Each prompt instructs the model to return a strict JSON object so the router can parse it
deterministically.
"""

from __future__ import annotations

import json

PARSE_SYSTEM = """You are a bank-statement parser. You receive a bank or card statement in \
any format (CSV, tabular text, HTML, copy-paste, or an image of a statement) in any language. \
Extract every transaction into a normalized JSON object.

Return ONLY a JSON object with this exact shape:
{
  "transactions": [
    {
      "date": "YYYY-MM-DD",
      "concept": "merchant or description, cleaned but faithful",
      "amount": -67.82,
      "balance": 1234.56,
      "transaction_type": "e.g. card payment, transfer, direct debit, Bizum, fee",
      "notes": ""
    }
  ]
}

Rules:
- amount is a number: negative for money out (charges, payments), positive for money in (income, refunds).
- Use a dot as the decimal separator. Convert "1.234,56" (EU format) to 1234.56.
- date MUST be ISO format YYYY-MM-DD. Infer the year from context if needed.
- balance is the running balance if present, else null.
- Do NOT invent transactions. If a field is unknown, use null (or "" for notes).
- Preserve the original language of the concept text. Do not translate it.
- Return the JSON object only — no markdown, no commentary."""

CATEGORIZE_SYSTEM = """You categorize a single bank transaction. You are given the transaction \
details and a fixed list of allowed category labels. Choose the SINGLE best matching label.

Return ONLY a JSON object:
{
  "category": "<one label copied EXACTLY from the allowed list, or null>",
  "confidence": 0.0,
  "suggested_category": null
}

Rules:
- "category" MUST be copied verbatim from the allowed list, or null if none fits. NEVER invent a
  label here that is not in the allowed list.
- "confidence" is your certainty from 0.0 to 1.0.
- Consider the merchant/concept, the amount sign, the transaction type and any notes.
- "suggested_category": usually null. Only when NO allowed label fits well AND a clearly better,
  more specific category would — propose a NEW one as
  { "name": "Short Name", "parent": "<an existing top-level label it would sit under, or null>",
  "reason": "<short reason>" }. Write BOTH "name" and "reason" in the user's language given as
  "language" in the user message ("es" = Spanish, "en" = English).
- NEVER propose a name listed as already rejected by the user (see the user message).
- Do not propose a new category that merely duplicates an existing allowed label.
- Return the JSON object only — no markdown, no commentary."""

ENRICH_SYSTEM = """You parse a purchase document into individual line items. You receive it as \
text or as an image. It may be a single receipt/ticket OR a screenshot of an online order history \
(e.g. Amazon) that contains SEVERAL independent purchases on different dates.

Return ONLY a JSON object:
{
  "items": [
    { "description": "product name as printed", "amount": 1.20, "category_suggestion": "Group / Subgroup" }
  ],
  "total_parsed": 0.00,
  "movements": null
}

Rules:
- amount is the line total for that item (quantity x unit price), as a positive number.
- category_suggestion is a short, human-readable hint like "Alimentación / Lácteos". It is a hint only.
- total_parsed is the sum of ALL the item amounts you extracted.
- Ignore subtotals, taxes lines, totals, change, and payment-method lines — only real products.
- Preserve the original language of the descriptions.
- "movements": usually null. It is a SINGLE purchase (a receipt/ticket, or one order) → leave it null
  and just fill "items". BUT if the document clearly shows SEVERAL INDEPENDENT purchases (an order
  history with different dates and/or order numbers), return them as a list, one object per purchase:
  { "date": "YYYY-MM-DD" or null, "concept": "merchant or order reference",
    "amount": <that purchase's own total, positive>,
    "items": [ { "description": ..., "amount": ..., "category_suggestion": ... }, ... ] }.
  Put each purchase's own products under its own movement, and still fill the flat "items" with all of them.
- Do NOT split a single multi-product purchase into several movements — only split genuinely separate orders.
- Return the JSON object only — no markdown, no commentary."""


def categorize_user_prompt(
    concept: str,
    amount: float,
    transaction_type: str | None,
    notes: str | None,
    categories: list[str],
    rejected_suggestions: list[str] | None = None,
    language: str = "en",
) -> str:
    payload = {
        "concept": concept,
        "amount": amount,
        "transaction_type": transaction_type,
        "notes": notes,
        "allowed_categories": categories,
        "rejected_new_categories": rejected_suggestions or [],
        "language": language if language in ("en", "es") else "en",
    }
    return (
        "Categorize this transaction. Allowed categories are in the JSON. Do not propose any name "
        "listed under rejected_new_categories.\n"
        + json.dumps(payload, ensure_ascii=False)
    )


def enrich_user_prompt(transaction_amount: float) -> str:
    return (
        "Parse the receipt into line items. For reference, the transaction total is "
        f"{transaction_amount:.2f} (negative means money spent). Extract the product lines."
    )
