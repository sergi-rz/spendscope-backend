from app.services import llm, revenuecat
from app.services.prompts import enrich_user_prompt
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


def test_enrich_returns_movements_for_order_history(client, monkeypatch):
    data = {
        "items": [
            {"description": "Pañales", "amount": 20.0, "category_suggestion": "Niños / Pañales"},
            {"description": "Libro", "amount": 15.0, "category_suggestion": "Ocio / Libros"},
        ],
        "total_parsed": 35.0,
        "movements": [
            {"date": "2026-05-02", "concept": "Amazon", "amount": 20.0,
             "items": [{"description": "Pañales", "amount": 20.0, "category_suggestion": "Niños / Pañales"}]},
            {"date": "2026-05-09", "concept": "Amazon", "amount": 15.0,
             "items": [{"description": "Libro", "amount": 15.0, "category_suggestion": "Ocio / Libros"}]},
        ],
    }
    monkeypatch.setattr(enrich_router.revenuecat, "is_premium", _premium(True))
    monkeypatch.setattr(enrich_router.llm, "complete_json", _fake(data))
    resp = client.post(
        "/api/v1/enrich",
        json={"user_id": "u1", "input_type": "text", "content": "order history", "transaction_amount": 0.0},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["movements"] is not None
    assert len(body["movements"]) == 2
    assert body["movements"][0]["date"] == "2026-05-02"
    assert body["movements"][0]["amount"] == 20.0
    assert len(body["movements"][0]["items"]) == 1


def test_enrich_single_receipt_has_null_movements(client, monkeypatch):
    monkeypatch.setattr(enrich_router.revenuecat, "is_premium", _premium(True))
    monkeypatch.setattr(enrich_router.llm, "complete_json", _fake(ITEMS))
    resp = client.post(
        "/api/v1/enrich",
        json={"user_id": "u1", "input_type": "text", "content": "t", "transaction_amount": -3.0},
    )
    assert resp.json()["movements"] is None


def test_enrich_flattens_items_from_movements_only(client, monkeypatch):
    data = {
        "movements": [
            {"date": None, "concept": "Amazon", "amount": 10.0,
             "items": [{"description": "X", "amount": 10.0, "category_suggestion": None}]},
        ],
    }
    monkeypatch.setattr(enrich_router.revenuecat, "is_premium", _premium(True))
    monkeypatch.setattr(enrich_router.llm, "complete_json", _fake(data))
    resp = client.post(
        "/api/v1/enrich",
        json={"user_id": "u1", "input_type": "text", "content": "order history", "transaction_amount": 0.0},
    )
    body = resp.json()
    assert len(body["items"]) == 1
    assert body["total_parsed"] == 10.0


def test_enrich_returns_ticket_total_and_date(client, monkeypatch):
    # #51: the receipt's printed total/subtotal/date come through so the app can anchor + confirm.
    data = {
        "items": [{"description": "A", "amount": 1.20}, {"description": "B", "amount": 1.80}],
        "total_parsed": 3.00,
        "ticket_total": 211.58,
        "subtotal": 219.64,
        "ticket_date": "2026-04-29",
    }
    monkeypatch.setattr(enrich_router.revenuecat, "is_premium", _premium(True))
    monkeypatch.setattr(enrich_router.llm, "complete_json", _fake(data))
    resp = client.post(
        "/api/v1/enrich",
        json={"user_id": "u1", "input_type": "text", "content": "t", "transaction_amount": 0.0},
    )
    body = resp.json()
    assert body["ticket_total"] == 211.58
    assert body["subtotal"] == 219.64
    assert body["ticket_date"] == "2026-04-29"


def test_enrich_matches_uses_printed_total_over_item_sum(client, monkeypatch):
    # Items sum to 3.00 but the receipt was paid 211.58; matching the known statement amount must
    # use the printed ticket_total, not the (incomplete) item sum.
    data = {
        "items": [{"description": "A", "amount": 1.20}, {"description": "B", "amount": 1.80}],
        "total_parsed": 3.00,
        "ticket_total": 211.58,
    }
    monkeypatch.setattr(enrich_router.revenuecat, "is_premium", _premium(True))
    monkeypatch.setattr(enrich_router.llm, "complete_json", _fake(data))
    resp = client.post(
        "/api/v1/enrich",
        json={"user_id": "u1", "input_type": "text", "content": "t", "transaction_amount": -211.58},
    )
    assert resp.json()["matches_transaction"] is True


def test_completeness_gate_rejects_short_item_sum():
    # Sum 141.76 vs printed subtotal 219.64 → far off → escalate.
    assert enrich_router._completeness_ok(
        {"items": [{"description": "x", "amount": 141.76}], "subtotal": 219.64}
    ) is False


def test_completeness_gate_accepts_matching_sum():
    assert enrich_router._completeness_ok(
        {"items": [{"description": "x", "amount": 100.0}, {"description": "y", "amount": 119.64}],
         "subtotal": 219.64}
    ) is True


def test_completeness_gate_accepts_when_nothing_to_check():
    # No printed subtotal/total → we can't judge completeness → accept (don't escalate blindly).
    assert enrich_router._completeness_ok(
        {"items": [{"description": "x", "amount": 5.0}]}
    ) is True


def test_completeness_gate_rejects_empty_items():
    assert enrich_router._completeness_ok({"items": [], "subtotal": 50.0}) is False


def test_parse_items_drops_non_product_lines():
    # #54: totals, VAT table, loyalty, coupons and bare numbers must not leak into the breakdown.
    raw = [
        {"description": "Leche 1L", "amount": 1.20},
        {"description": "SUBTOTAL", "amount": 219.64},
        {"description": "TOTAL A PAGAR", "amount": 211.58},
        {"description": "4,00%", "amount": 45.56},
        {"description": "45,56", "amount": 45.56},
        {"description": "DTO. CUPON 6X50 EUR", "amount": 6.00},
        {"description": "VENTAJAS OBTENIDAS", "amount": 16.28},
        {"description": "2 x ( 2,09 )", "amount": 4.18},
        {"description": "Pan", "amount": 1.80},
    ]
    items = enrich_router._parse_items(raw)
    assert [i.description for i in items] == ["Leche 1L", "Pan"]


def test_flatten_items_drops_amount_equal_to_total(client):
    # Even with a product-looking description, an amount equal to the printed total is the total.
    data = {
        "items": [
            {"description": "Compra", "amount": 211.58},
            {"description": "Leche", "amount": 1.20},
        ],
        "ticket_total": 211.58,
        "subtotal": 219.64,
    }
    items = enrich_router._flatten_items(data)
    assert [i.description for i in items] == ["Leche"]


def test_enrich_prompt_includes_allowed_categories():
    # #55: the enricher must pick from the real taxonomy so suggestions resolve in the app.
    prompt = enrich_user_prompt(-211.58, ["Alimentación / Fruta y verdura", "Hogar / Limpieza"])
    assert "allowed_categories" in prompt
    assert "Alimentación / Fruta y verdura" in prompt
    assert "VERBATIM" in prompt


def test_enrich_prompt_free_form_without_categories():
    prompt = enrich_user_prompt(-10.0, [])
    assert "allowed_categories" not in prompt


def test_enrich_forwards_categories_to_prompt(client, monkeypatch):
    captured = {}

    async def fake(system, user_text, image_data_uri=None, accept=None):
        captured["user_text"] = user_text
        return llm.LLMResult(data=ITEMS, provider_used="nan", is_fallback=False, primary_error=None)

    monkeypatch.setattr(enrich_router.revenuecat, "is_premium", _premium(True))
    monkeypatch.setattr(enrich_router.llm, "complete_json", fake)
    resp = client.post(
        "/api/v1/enrich",
        json={"user_id": "u1", "input_type": "text", "content": "t", "transaction_amount": -3.0,
              "categories": ["Alimentación / Carne", "Mascota"]},
    )
    assert resp.status_code == 200
    assert "Alimentación / Carne" in captured["user_text"]


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
