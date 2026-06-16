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


async def test_bad_output_reports_finish_reason(monkeypatch):
    # A truncated JSON body (finish_reason=length) — what a too-large statement produces.
    payload = {"choices": [{"message": {"content": '{"transactions": [{"a": 1}'}, "finish_reason": "length"}]}

    async def fake_post(self, url, json=None, headers=None):
        return httpx.Response(200, json=payload, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
    with pytest.raises(llm.LLMBadOutput) as excinfo:
        await llm._call_provider(_provider("nan:gemma4"), system="s", user_text="u", image_data_uri=None)
    assert "finish_reason=length" in str(excinfo.value)


def test_short_error_keeps_bad_output_message():
    msg = llm._short_error(llm.LLMBadOutput("could not parse JSON: x [finish_reason=length, 4096 chars]"))
    assert msg.startswith("LLMBadOutput:")
    assert "finish_reason=length" in msg


def test_expand_single_model_keeps_plain_name():
    providers = llm._expand("nan_builders", "https://x/v1", "k", "gemma4", "gemma4")
    assert len(providers) == 1
    assert providers[0].name == "nan_builders"
    assert providers[0].model_text == "gemma4"


def test_expand_model_chain_into_tiers():
    providers = llm._expand("nan_builders", "https://x/v1", "k", "gemma4,qwen3.6", "gemma4,qwen3.6")
    assert [p.name for p in providers] == ["nan_builders:gemma4", "nan_builders:qwen3.6"]
    assert [p.model_text for p in providers] == ["gemma4", "qwen3.6"]
    assert [p.model_vision for p in providers] == ["gemma4", "qwen3.6"]


def test_expand_pads_shorter_list_with_last():
    # Two text models, one vision model → vision reuses its single entry for both tiers.
    providers = llm._expand("nan_builders", "https://x/v1", "k", "gemma4,qwen3.6", "gemma4")
    assert [p.model_vision for p in providers] == ["gemma4", "gemma4"]


async def test_three_tier_chain_walks_in_order(monkeypatch):
    monkeypatch.setattr(
        llm,
        "_providers",
        lambda: [_provider("nan:gemma4"), _provider("nan:qwen3.6"), _provider("openai")],
    )
    calls = []

    async def fake_call(provider, system, user_text, image):
        calls.append(provider.name)
        if provider.name != "openai":
            raise httpx.ConnectError("boom")
        return {"ok": True}

    monkeypatch.setattr(llm, "_call_provider", fake_call)

    result = await llm.complete_json(system="s", user_text="u")
    assert result.provider_used == "openai"
    assert calls == ["nan:gemma4", "nan:qwen3.6", "openai"]
