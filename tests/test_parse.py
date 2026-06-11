import pytest

from app.services import llm
from app.routers import parse as parse_router


def _fake_result(data, provider="nan_builders", fallback=False, primary_error=None):
    async def _fake(*args, **kwargs):
        return llm.LLMResult(
            data=data, provider_used=provider, is_fallback=fallback, primary_error=primary_error
        )

    return _fake


SAMPLE = {
    "transactions": [
        {
            "date": "2026-05-01",
            "concept": "MERCADONA",
            "amount": -67.82,
            "balance": 2450.18,
            "transaction_type": "Pago con tarjeta",
            "notes": "",
        },
        {
            "date": "2026-05-02",
            "concept": "NOMINA",
            "amount": 2100.0,
            "balance": 4550.18,
            "transaction_type": "Transferencia",
            "notes": "mayo",
        },
    ],
    "bank_detected": "BBVA",
}


def test_parse_text_ok(client, monkeypatch):
    monkeypatch.setattr(parse_router.llm, "complete_json", _fake_result(SAMPLE))
    resp = client.post(
        "/api/v1/parse",
        json={"user_id": "u1", "input_type": "text", "content": "csv...", "filename": "x.csv"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["metadata"]["count"] == 2
    assert body["metadata"]["bank_detected"] == "BBVA"
    first = body["transactions"][0]
    # snake_case keys, exactly what the app decodes.
    assert set(first) == {"date", "concept", "amount", "balance", "transaction_type", "notes"}
    assert first["concept"] == "MERCADONA"
    assert first["amount"] == -67.82


def test_parse_skips_malformed_rows(client, monkeypatch):
    data = {"transactions": [{"date": "2026-05-01", "concept": "OK", "amount": -1.0}, {"bad": True}]}
    monkeypatch.setattr(parse_router.llm, "complete_json", _fake_result(data))
    resp = client.post(
        "/api/v1/parse", json={"user_id": "u1", "input_type": "text", "content": "x"}
    )
    assert resp.status_code == 200
    assert resp.json()["metadata"]["count"] == 1


def test_parse_provider_unavailable_returns_502(client, monkeypatch):
    async def _boom(*args, **kwargs):
        raise llm.LLMUnavailable("all providers failed")

    monkeypatch.setattr(parse_router.llm, "complete_json", _boom)
    resp = client.post(
        "/api/v1/parse", json={"user_id": "u1", "input_type": "text", "content": "x"}
    )
    assert resp.status_code == 502


def test_parse_requires_known_fields(client):
    resp = client.post("/api/v1/parse", json={"input_type": "text", "content": "x"})
    assert resp.status_code == 422  # missing user_id
