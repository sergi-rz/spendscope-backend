"""Per-request cost estimation for operational logging (#bench).

Only the OpenAI fallback has a public per-token price; gemma4/qwen run on nan.builders (EU,
flat/self-hosted) so they cost 0 per token here. We log a USD estimate per request so we can
later aggregate monthly spend and decide routing (e.g. budget-gated provider switching).

Prices in USD per 1M tokens (input, output). Update when OpenAI changes pricing.
"""

from __future__ import annotations

MODEL_PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1-mini": (0.40, 1.60),
}


def cost_usd(model: str | None, in_tokens: int, out_tokens: int) -> float:
    """USD estimate for one call. Returns 0.0 for models without a per-token price (gemma/qwen)."""
    rate = MODEL_PRICING.get((model or "").strip().lower())
    if not rate:
        return 0.0
    return in_tokens / 1_000_000 * rate[0] + out_tokens / 1_000_000 * rate[1]
