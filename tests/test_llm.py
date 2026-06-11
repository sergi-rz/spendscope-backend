import httpx
import pytest

from app.services import llm
from app.services.llm import Provider, _extract_json


def test_extract_plain_json():
    assert _extract_json('{"a": 1}') == {"a": 1}


def test_extract_fenced_json():
    assert _extract_json('```json\n{"a": 1}\n```') == {"a": 1}


def test_extract_embedded_json():
    assert _extract_json('Sure! {"a": 1} done') == {"a": 1}


def test_extract_no_json_raises():
    with pytest.raises(llm.LLMBadOutput):
        _extract_json("no json here")


def _provider(name):
    return Provider(name=name, base_url="https://x/v1", api_key="k", model_text="m", model_vision="mv")


async def test_falls_back_to_second_provider(monkeypatch):
    monkeypatch.setattr(llm, "_providers", lambda: [_provider("nan_builders"), _provider("openai")])

    calls = []

    async def fake_call(provider, system, user_text, image):
        calls.append(provider.name)
        if provider.name == "nan_builders":
            raise httpx.ConnectError("boom")
        return {"ok": True}

    monkeypatch.setattr(llm, "_call_provider", fake_call)

    result = await llm.complete_json(system="s", user_text="u")
    assert result.provider_used == "openai"
    assert result.is_fallback is True
    assert result.primary_error  # recorded why primary failed
    assert calls == ["nan_builders", "openai"]


async def test_unavailable_when_no_provider_configured(monkeypatch):
    monkeypatch.setattr(llm, "_providers", lambda: [])
    with pytest.raises(llm.LLMUnavailable):
        await llm.complete_json(system="s", user_text="u")
