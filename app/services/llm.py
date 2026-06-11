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


def _providers() -> list[Provider]:
    return [
        Provider(
            name="nan_builders",
            base_url=settings.primary_base_url,
            api_key=settings.primary_api_key,
            model_text=settings.primary_model_text,
            model_vision=settings.primary_model_vision,
        ),
        Provider(
            name="openai",
            base_url=settings.fallback_base_url,
            api_key=settings.fallback_api_key,
            model_text=settings.fallback_model_text,
            model_vision=settings.fallback_model_vision,
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
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMBadOutput(f"{provider.name} response missing content") from exc
    return _extract_json(content)


async def complete_json(
    *, system: str, user_text: str, image_data_uri: str | None = None
) -> LLMResult:
    """Run the prompt through the provider chain and return parsed JSON + provider metadata."""
    providers = [p for p in _providers() if p.configured]
    if not providers:
        raise LLMUnavailable("no LLM provider configured (set PRIMARY_API_KEY / FALLBACK_API_KEY)")

    primary_error: str | None = None
    for index, provider in enumerate(providers):
        try:
            data = await _call_provider(provider, system, user_text, image_data_uri)
            return LLMResult(
                data=data,
                provider_used=provider.name,
                is_fallback=index > 0,
                primary_error=primary_error,
            )
        except (httpx.HTTPError, LLMBadOutput) as exc:
            message = str(exc)
            logger.warning("provider %s failed: %s", provider.name, message)
            if index == 0:
                primary_error = _short_error(exc)
            continue

    raise LLMUnavailable(primary_error or "all providers failed")


def _short_error(exc: Exception) -> str:
    if isinstance(exc, httpx.TimeoutException):
        return "timeout"
    if isinstance(exc, httpx.HTTPStatusError):
        return str(exc.response.status_code)
    return type(exc).__name__
