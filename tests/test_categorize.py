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


def test_categorize_forwards_language_to_prompt(client, monkeypatch):
    captured = {}

    async def _capture(*args, **kwargs):
        captured["user_text"] = kwargs.get("user_text")
        return llm.LLMResult(
            data={"category": None, "confidence": 0.0}, provider_used="x", is_fallback=False, primary_error=None
        )

    monkeypatch.setattr(cat_router.llm, "complete_json", _capture)
    resp = client.post(
        "/api/v1/categorize",
        json={"user_id": "u1", "concept": "X", "amount": -1.0, "categories": CATEGORIES, "language": "es"},
    )
    assert resp.status_code == 200
    assert '"language": "es"' in captured["user_text"]


def test_categorize_language_defaults_to_en(client, monkeypatch):
    captured = {}

    async def _capture(*args, **kwargs):
        captured["user_text"] = kwargs.get("user_text")
        return llm.LLMResult(
            data={"category": None, "confidence": 0.0}, provider_used="x", is_fallback=False, primary_error=None
        )

    monkeypatch.setattr(cat_router.llm, "complete_json", _capture)
    client.post(
        "/api/v1/categorize",
        json={"user_id": "u1", "concept": "X", "amount": -1.0, "categories": CATEGORIES},
    )
    assert '"language": "en"' in captured["user_text"]


# --- POST /categorize/batch (#44) ---------------------------------------------


def test_categorize_batch_maps_results_by_index(client, monkeypatch):
    data = {"results": [
        {"index": 0, "category": "Restaurantes / Cafés", "confidence": 0.9},
        {"index": 1, "category": "Alimentación / Supermercado", "confidence": 0.8},
    ]}
    monkeypatch.setattr(cat_router.llm, "complete_json", _fake(data))
    resp = client.post(
        "/api/v1/categorize/batch",
        json={
            "user_id": "u1",
            "items": [
                {"concept": "BATCH BAR", "amount": -5.0},
                {"concept": "BATCH MERCADONA", "amount": -30.0},
            ],
            "categories": CATEGORIES,
        },
    )
    assert resp.status_code == 200
    results = resp.json()["results"]
    assert [r["index"] for r in results] == [0, 1]
    assert results[0]["category"] == "Restaurantes / Cafés"
    assert results[1]["category"] == "Alimentación / Supermercado"


def test_categorize_batch_one_call_for_many_items(client, monkeypatch):
    calls = {"n": 0}

    async def _count(*args, **kwargs):
        calls["n"] += 1
        return llm.LLMResult(
            data={"results": [{"index": i, "category": None} for i in range(5)]},
            provider_used="x", is_fallback=False, primary_error=None,
        )

    monkeypatch.setattr(cat_router.llm, "complete_json", _count)
    items = [{"concept": f"BATCHN {i}", "amount": -float(i + 1)} for i in range(5)]
    resp = client.post(
        "/api/v1/categorize/batch", json={"user_id": "u1", "items": items, "categories": CATEGORIES}
    )
    assert resp.status_code == 200
    assert len(resp.json()["results"]) == 5
    assert calls["n"] == 1, "the whole batch must be a single LLM call"


def test_categorize_batch_forwards_source_category_and_skips_cache(client, monkeypatch):
    # An imported item carries the user's category from another app (#66): it must reach the prompt
    # as a strong hint, and it must NOT be served from / written to the concept cache.
    captured = {}
    calls = {"n": 0}

    async def _capture(*args, **kwargs):
        calls["n"] += 1
        captured["user_text"] = kwargs.get("user_text")
        return llm.LLMResult(
            data={"results": [{"index": 0, "category": CATEGORIES[0], "confidence": 0.95}]},
            provider_used="x", is_fallback=False, primary_error=None,
        )

    monkeypatch.setattr(cat_router.llm, "complete_json", _capture)
    payload = {
        "user_id": "u1",
        "items": [{"concept": "SRCCAT UNIQUE", "amount": -3.0, "source_category": "Coffee Shops"}],
        "categories": CATEGORIES,
    }
    first = client.post("/api/v1/categorize/batch", json=payload)
    assert first.status_code == 200
    assert "Coffee Shops" in captured["user_text"]
    assert first.json()["results"][0]["category"] == CATEGORIES[0]

    # Same concept again, hinted again, must hit the LLM again (no cache shortcut).
    client.post("/api/v1/categorize/batch", json=payload)
    assert calls["n"] == 2


def test_categorize_batch_forwards_already_suggested(client, monkeypatch):
    # #dedup: names proposed in earlier chunks of the same import must reach the prompt so the model
    # reuses them verbatim instead of coining a synonym per chunk.
    captured = {}

    async def _capture(*args, **kwargs):
        captured["user_text"] = kwargs.get("user_text")
        return llm.LLMResult(
            data={"results": [{"index": 0, "category": None}]},
            provider_used="x", is_fallback=False, primary_error=None,
        )

    monkeypatch.setattr(cat_router.llm, "complete_json", _capture)
    resp = client.post(
        "/api/v1/categorize/batch",
        json={
            "user_id": "u1",
            "items": [{"concept": "PELUQUERIA LOLA", "amount": -20.0}],
            "categories": CATEGORIES,
            "already_suggested": ["Peluquería"],
        },
    )
    assert resp.status_code == 200
    assert "already_suggested_categories" in captured["user_text"]
    assert "Peluquería" in captured["user_text"]


def test_categorize_batch_unknown_label_is_null(client, monkeypatch):
    monkeypatch.setattr(
        cat_router.llm, "complete_json", _fake({"results": [{"index": 0, "category": "Nope"}]})
    )
    resp = client.post(
        "/api/v1/categorize/batch",
        json={"user_id": "u1", "items": [{"concept": "BATCH X", "amount": -1.0}], "categories": CATEGORIES},
    )
    assert resp.json()["results"][0]["category"] is None


def test_categorize_batch_empty_items_returns_empty(client):
    resp = client.post(
        "/api/v1/categorize/batch",
        json={"user_id": "u1", "items": [], "categories": CATEGORIES},
    )
    assert resp.status_code == 200
    assert resp.json()["results"] == []


def test_categorize_batch_empty_categories_400(client):
    resp = client.post(
        "/api/v1/categorize/batch",
        json={"user_id": "u1", "items": [{"concept": "X", "amount": -1.0}], "categories": []},
    )
    assert resp.status_code == 400


def test_categorize_batch_graceful_when_llm_down(client, monkeypatch):
    async def _boom(*args, **kwargs):
        raise llm.LLMUnavailable("down")

    monkeypatch.setattr(cat_router.llm, "complete_json", _boom)
    resp = client.post(
        "/api/v1/categorize/batch",
        json={"user_id": "u1", "items": [{"concept": "BATCH DOWN", "amount": -1.0}], "categories": CATEGORIES},
    )
    assert resp.status_code == 200  # partial success, never blocks the import
    assert resp.json()["results"][0]["category"] is None


def test_categorize_returns_suggested_pattern(client, monkeypatch):
    monkeypatch.setattr(
        cat_router.llm,
        "complete_json",
        _fake({"category": CATEGORIES[0], "confidence": 0.9, "suggested_pattern": "Amazon"}),
    )
    resp = client.post(
        "/api/v1/categorize",
        json={"user_id": "u1", "concept": "www.amazon.esv1332423", "amount": -19.9, "categories": CATEGORIES},
    )
    assert resp.json()["suggested_pattern"] == "amazon"  # lowercased, reusable token


def test_categorize_drops_hallucinated_pattern(client, monkeypatch):
    # The model returns a token that is NOT in the concept — we must not trust it (#50).
    monkeypatch.setattr(
        cat_router.llm,
        "complete_json",
        _fake({"category": CATEGORIES[0], "confidence": 0.9, "suggested_pattern": "amazon"}),
    )
    resp = client.post(
        "/api/v1/categorize",
        json={"user_id": "u1", "concept": "MERCADONA 1234", "amount": -19.9, "categories": CATEGORIES},
    )
    assert resp.json()["suggested_pattern"] is None


def test_categorize_batch_returns_suggested_pattern(client, monkeypatch):
    data = {"results": [
        {"index": 0, "category": CATEGORIES[0], "confidence": 0.8, "suggested_pattern": "MERCADONA"},
    ]}
    monkeypatch.setattr(cat_router.llm, "complete_json", _fake(data))
    resp = client.post(
        "/api/v1/categorize/batch",
        json={"user_id": "u1", "items": [{"concept": "COMPRA TARJ 4587 MERCADONA 1234", "amount": -30.0}],
              "categories": CATEGORIES},
    )
    assert resp.json()["results"][0]["suggested_pattern"] == "mercadona"
