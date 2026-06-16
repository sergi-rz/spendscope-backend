from app.services import llm
from app.routers import categorize as cat_router

CATEGORIES = ["Alimentación / Supermercado", "Restaurantes / Cafés", "Ocio / Videojuegos"]


def _fake(data):
    async def _f(*args, **kwargs):
        return llm.LLMResult(data=data, provider_used="nan_builders", is_fallback=False, primary_error=None)

    return _f


def test_categorize_returns_exact_label(client, monkeypatch):
    monkeypatch.setattr(
        cat_router.llm, "complete_json", _fake({"category": "Restaurantes / Cafés", "confidence": 0.9})
    )
    resp = client.post(
        "/api/v1/categorize",
        json={"user_id": "u1", "concept": "BAR LA ESQUINA", "amount": -12.5, "categories": CATEGORIES},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["category"] == "Restaurantes / Cafés"
    assert body["confidence"] == 0.9


def test_categorize_label_case_insensitive_resolves_to_canonical(client, monkeypatch):
    monkeypatch.setattr(
        cat_router.llm, "complete_json", _fake({"category": "restaurantes / cafés", "confidence": 0.7})
    )
    resp = client.post(
        "/api/v1/categorize",
        json={"user_id": "u1", "concept": "BAR", "amount": -5.0, "categories": CATEGORIES},
    )
    assert resp.json()["category"] == "Restaurantes / Cafés"  # canonical casing from the allowed list


def test_categorize_unknown_label_returns_null(client, monkeypatch):
    monkeypatch.setattr(
        cat_router.llm, "complete_json", _fake({"category": "Something Else", "confidence": 0.4})
    )
    resp = client.post(
        "/api/v1/categorize",
        json={"user_id": "u1", "concept": "???", "amount": -5.0, "categories": CATEGORIES},
    )
    assert resp.status_code == 200
    assert resp.json()["category"] is None


def test_categorize_empty_categories_400(client):
    resp = client.post(
        "/api/v1/categorize",
        json={"user_id": "u1", "concept": "X", "amount": -1.0, "categories": []},
    )
    assert resp.status_code == 400


def test_categorize_confidence_clamped(client, monkeypatch):
    monkeypatch.setattr(
        cat_router.llm, "complete_json", _fake({"category": CATEGORIES[0], "confidence": 1.8})
    )
    resp = client.post(
        "/api/v1/categorize",
        json={"user_id": "u1", "concept": "MERCADONA", "amount": -10.0, "categories": CATEGORIES},
    )
    assert resp.json()["confidence"] == 1.0


def test_categorize_returns_new_category_suggestion(client, monkeypatch):
    monkeypatch.setattr(
        cat_router.llm,
        "complete_json",
        _fake({
            "category": None,
            "confidence": 0.2,
            "suggested_category": {"name": "Parking", "parent": "Transporte", "reason": "garaje"},
        }),
    )
    resp = client.post(
        "/api/v1/categorize",
        json={"user_id": "u1", "concept": "SABA APARCAMIENTO", "amount": -3.2, "categories": CATEGORIES},
    )
    body = resp.json()
    assert body["category"] is None
    assert body["suggested_category"]["name"] == "Parking"
    assert body["suggested_category"]["parent"] == "Transporte"


def test_categorize_ignores_suggestion_duplicating_existing(client, monkeypatch):
    monkeypatch.setattr(
        cat_router.llm,
        "complete_json",
        _fake({"category": None, "suggested_category": {"name": "Cafés"}}),  # leaf of an allowed label
    )
    resp = client.post(
        "/api/v1/categorize",
        json={"user_id": "u1", "concept": "X", "amount": -1.0, "categories": CATEGORIES},
    )
    assert resp.json()["suggested_category"] is None


def test_categorize_ignores_rejected_suggestion(client, monkeypatch):
    monkeypatch.setattr(
        cat_router.llm,
        "complete_json",
        _fake({"category": None, "suggested_category": {"name": "Parking"}}),
    )
    resp = client.post(
        "/api/v1/categorize",
        json={
            "user_id": "u1", "concept": "X", "amount": -1.0, "categories": CATEGORIES,
            "rejected_suggestions": ["parking"],
        },
    )
    assert resp.json()["suggested_category"] is None


def test_categorize_no_suggestion_field_is_null(client, monkeypatch):
    monkeypatch.setattr(
        cat_router.llm, "complete_json", _fake({"category": "Restaurantes / Cafés", "confidence": 0.9})
    )
    resp = client.post(
        "/api/v1/categorize",
        json={"user_id": "u1", "concept": "BAR", "amount": -5.0, "categories": CATEGORIES},
    )
    assert resp.json()["suggested_category"] is None
