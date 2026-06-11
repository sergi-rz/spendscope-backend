from app.services import llm, revenuecat
from app.routers import enrich as enrich_router


def _fake(data):
    async def _f(*args, **kwargs):
        return llm.LLMResult(data=data, provider_used="nan_builders", is_fallback=False, primary_error=None)

    return _f


def _premium(value):
    async def _p(user_id):
        return value

    return _p


ITEMS = {
    "items": [
        {"description": "Leche 1L", "amount": 1.20, "category_suggestion": "Alimentación / Lácteos"},
        {"description": "Pan", "amount": 1.80, "category_suggestion": "Alimentación / Panadería"},
    ],
    "total_parsed": 3.00,
}


def test_enrich_premium_ok_and_matches(client, monkeypatch):
    monkeypatch.setattr(enrich_router.revenuecat, "is_premium", _premium(True))
    monkeypatch.setattr(enrich_router.llm, "complete_json", _fake(ITEMS))
    resp = client.post(
        "/api/v1/enrich",
        json={"user_id": "u1", "input_type": "text", "content": "ticket...", "transaction_amount": -3.0},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["items"]) == 2
    assert body["total_parsed"] == 3.0
    assert body["matches_transaction"] is True
    assert set(body["items"][0]) == {"description", "amount", "category_suggestion"}


def test_enrich_total_mismatch(client, monkeypatch):
    monkeypatch.setattr(enrich_router.revenuecat, "is_premium", _premium(True))
    monkeypatch.setattr(enrich_router.llm, "complete_json", _fake(ITEMS))
    resp = client.post(
        "/api/v1/enrich",
        json={"user_id": "u1", "input_type": "text", "content": "t", "transaction_amount": -50.0},
    )
    assert resp.json()["matches_transaction"] is False


def test_enrich_computes_total_when_missing(client, monkeypatch):
    data = {"items": [{"description": "A", "amount": 2.0}, {"description": "B", "amount": 3.5}]}
    monkeypatch.setattr(enrich_router.revenuecat, "is_premium", _premium(True))
    monkeypatch.setattr(enrich_router.llm, "complete_json", _fake(data))
    resp = client.post(
        "/api/v1/enrich",
        json={"user_id": "u1", "input_type": "text", "content": "t", "transaction_amount": -5.5},
    )
    body = resp.json()
    assert body["total_parsed"] == 5.5
    assert body["matches_transaction"] is True


def test_enrich_not_premium_403(client, monkeypatch):
    monkeypatch.setattr(enrich_router.revenuecat, "is_premium", _premium(False))
    resp = client.post(
        "/api/v1/enrich",
        json={"user_id": "u1", "input_type": "text", "content": "t", "transaction_amount": -3.0},
    )
    assert resp.status_code == 403


def test_enrich_premium_unavailable_503(client, monkeypatch):
    async def _raise(user_id):
        raise revenuecat.PremiumCheckUnavailable("no key")

    monkeypatch.setattr(enrich_router.revenuecat, "is_premium", _raise)
    resp = client.post(
        "/api/v1/enrich",
        json={"user_id": "u1", "input_type": "text", "content": "t", "transaction_amount": -3.0},
    )
    assert resp.status_code == 503
