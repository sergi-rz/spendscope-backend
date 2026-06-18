"""LLM provider router (SPECS §4.2, §11.4).

Tries the primary provider (nan.builders, EU) first and falls back to OpenAI on any error.
Both are OpenAI-compatible Chat Completions endpoints, so a single code path serves both.
Returns parsed JSON plus the metadata needed for operational logging (which provider answered,
whether it was a fallback, and why the primary failed).

No inputs or outputs are persisted here — process and discard (SPECS §7.4).
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass

import httpx

from ..config import settings

logger = logging.getLogger("spendscope.llm")

_JSON_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


class LLMUnavailable(Exception):
    """Every configured provider failed."""


class LLMBadOutput(Exception):
    """A provider answered but the content wasn't usable JSON."""


@dataclass
class Provider:
    name: str
    base_url: str
    api_key: str
    model_text: str
    model_vision: str

    @property
    def configured(self) -> bool:
        return bool(self.api_key and self.base_url)


def _split_models(value: str) -> list[str]:
    """A credential set may list several models (comma-separated), tried in order."""
    return [m.strip() for m in value.split(",") if m.strip()]


def _expand(
    cred_name: str, base_url: str, api_key: str, model_text: str, model_vision: str
) -> list[Provider]:
    """Expand a credential set into one Provider per model tier.

    `PRIMARY_MODEL_TEXT=gemma4,qwen3.6` becomes two tiers (gemma4 first, qwen3.6 next),
    both sharing the same base_url/api_key. Text and vision model lists are paired by
    index; a shorter list reuses its last entry so the counts need not match.
    """
    text_list = _split_models(model_text)
    vision_list = _split_models(model_vision)
    tiers = max(len(text_list), len(vision_list))
    providers: list[Provider] = []
    for i in range(tiers):
        mt = text_list[i] if i < len(text_list) else (text_list[-1] if text_list else "")
        mv = vision_list[i] if i < len(vision_list) else (vision_list[-1] if vision_list else "")
        label = mt or mv or cred_name
        providers.append(
            Provider(
                name=f"{cred_name}:{label}" if tiers > 1 else cred_name,
                base_url=base_url,
                api_key=api_key,
                model_text=mt,
                model_vision=mv,
            )
        )
    return providers


def _providers() -> list[Provider]:
    """Ordered provider chain (SPECS §11.5). N tiers: each credential set may expose
    several models, e.g. nan.builders `gemma4 → qwen3.6`, then OpenAI `gpt-4o-mini`."""
    return [
        *_expand(
            "nan_builders",
            settings.primary_base_url,
            settings.primary_api_key,
            settings.primary_model_text,
            settings.primary_model_vision,
        ),
        *_expand(
            "openai",
            settings.fallback_base_url,
            settings.fallback_api_key,
            settings.fallback_model_text,
            settings.fallback_model_vision,
        ),
    ]


@dataclass
class LLMResult:
    data: dict
    provider_used: str
    is_fallback: bool
    primary_error: str | None


def _build_messages(system: str, user_text: str, image_data_uri: str | None) -> list[dict]:
    if image_data_uri:
        user_content = [
            {"type": "text", "text": user_text},
            {"type": "image_url", "image_url": {"url": image_data_uri}},
        ]
    else:
        user_content = user_text
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]


def _extract_json(content: str) -> dict:
    stripped = _JSON_FENCE.sub("", content.strip())
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    # Last resort: grab the outermost { ... } block.
    start, end = stripped.find("{"), stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(stripped[start : end + 1])
        except json.JSONDecodeError as exc:
            raise LLMBadOutput(f"could not parse JSON: {exc}") from exc
    raise LLMBadOutput("response contained no JSON object")


async def _call_provider(
    provider: Provider, system: str, user_text: str, image_data_uri: str | None
) -> dict:
    model = provider.model_vision if image_data_uri else provider.model_text
    body = {
        "model": model,
        "messages": _build_messages(system, user_text, image_data_uri),
        "temperature": 0,
        "max_tokens": settings.llm_max_output_tokens,
        "response_format": {"type": "json_object"},
    }
    headers = {
        "Authorization": f"Bearer {provider.api_key}",
        "Content-Type": "application/json",
    }
    url = provider.base_url.rstrip("/") + "/chat/completions"

    async with httpx.AsyncClient(timeout=settings.llm_timeout_seconds) as client:
        resp = await client.post(url, json=body, headers=headers)

    if resp.status_code >= 400:
        raise httpx.HTTPStatusError(
            f"{provider.name} returned {resp.status_code}: {resp.text[:200]}",
            request=resp.request,
            response=resp,
        )

    payload = resp.json()
    try:
        choice = payload["choices"][0]
        content = choice["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMBadOutput(f"{provider.name} response missing content") from exc

    finish_reason = choice.get("finish_reason") if isinstance(choice, dict) else None
    try:
        return _extract_json(content)
    except LLMBadOutput as exc:
        # finish_reason="length" means the JSON was cut off by max_tokens (statement too big for one
        # call), not malformed — surfacing it makes LLMBadOutput in the logs/DB actionable.
        if settings.llm_debug_raw:
            logger.warning(
                "%s bad output (finish_reason=%s):\n%s",
                provider.name, finish_reason, (content or "")[: settings.llm_debug_raw_chars],
            )
        raise LLMBadOutput(
            f"{exc} [finish_reason={finish_reason}, {len(content or '')} chars]"
        ) from exc


async def complete_json(
    *,
    system: str,
    user_text: str,
    image_data_uri: str | None = None,
    accept: Callable[[dict], bool] | None = None,
) -> LLMResult:
    """Run the prompt through the provider chain and return parsed JSON + provider metadata.

    `accept` is an optional quality gate (#52). When given, a provider's parsed JSON is only
    returned if `accept(data)` is True; otherwise it is a *soft* rejection — the parse was valid
    but not good enough (e.g. an OCR pass whose item sum is far from the printed total) — so we
    escalate to the next, stronger provider. If no provider satisfies the gate, the strongest
    model's parse is returned as a best effort rather than failing.
    """
    providers = [p for p in _providers() if p.configured]
    if not providers:
        raise LLMUnavailable("no LLM provider configured (set PRIMARY_API_KEY / FALLBACK_API_KEY)")

    primary_error: str | None = None
    soft_rejected: LLMResult | None = None
    for index, provider in enumerate(providers):
        try:
            data = await _call_provider(provider, system, user_text, image_data_uri)
        except (httpx.HTTPError, LLMBadOutput) as exc:
            message = str(exc)
            logger.warning("provider %s failed: %s", provider.name, message)
            if index == 0:
                primary_error = _short_error(exc)
            continue

        result = LLMResult(
            data=data,
            provider_used=provider.name,
            is_fallback=index > 0,
            primary_error=primary_error,
        )
        if accept is None or accept(data):
            return result
        # Valid JSON but it failed the caller's quality gate: keep the strongest such parse as a
        # fallback and escalate to the next provider.
        logger.info("provider %s parse rejected by quality gate; escalating", provider.name)
        soft_rejected = result

    if soft_rejected is not None:
        return soft_rejected
    raise LLMUnavailable(primary_error or "all providers failed")


def _short_error(exc: Exception) -> str:
    if isinstance(exc, httpx.TimeoutException):
        return "timeout"
    if isinstance(exc, httpx.HTTPStatusError):
        return str(exc.response.status_code)
    if isinstance(exc, LLMBadOutput):
        # Keep the message (incl. finish_reason / char count) — that's what makes it debuggable.
        return f"LLMBadOutput: {exc}"[:240]
    return type(exc).__name__
