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
    assert body["metadata"]["bank_detected"] is None  # ignored (#27)
    first = body["transactions"][0]
    # snake_case keys, exactly what the app decodes.
    assert set(first) == {
        "date", "concept", "amount", "balance", "transaction_type", "notes", "source_category",
    }
    assert first["concept"] == "MERCADONA"
    assert first["amount"] == -67.82


def test_parse_extracts_source_category(client, monkeypatch):
    # An export from another finance app already carries a category column (#66).
    data = {
        "transactions": [
            {"date": "2026-05-01", "concept": "STARBUCKS", "amount": -4.5,
             "source_category": "Food & Dining / Coffee Shops"},
            {"date": "2026-05-02", "concept": "RENT", "amount": -900.0},  # no category column
        ]
    }
    monkeypatch.setattr(parse_router.llm, "complete_json", _fake_result(data))
    resp = client.post(
        "/api/v1/parse",
        json={"user_id": "u1", "input_type": "text", "content": "csv...", "filename": "mint.csv"},
    )
    assert resp.status_code == 200
    rows = resp.json()["transactions"]
    assert rows[0]["source_category"] == "Food & Dining / Coffee Shops"
    assert rows[1]["source_category"] is None


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


def test_parse_chunks_large_statement_and_merges(client, monkeypatch):
    # A flattened statement above the line threshold must be split into several LLM calls whose
    # transactions are merged in order (#LLMBadOutput fix).
    calls: list[str] = []

    async def _fake(*args, **kwargs):
        user_text = kwargs.get("user_text", "")
        calls.append(user_text)
        idx = len(calls)
        return llm.LLMResult(
            data={"transactions": [
                {"date": "2026-05-01", "concept": f"ROW{idx}", "amount": -float(idx)},
            ]},
            provider_used="nan_builders", is_fallback=False, primary_error=None,
        )

    monkeypatch.setattr(parse_router.llm, "complete_json", _fake)

    # 250 data lines with a header → more than one batch of parse_chunk_size.
    header = "Fecha\tConcepto\tImporte"
    rows = "\n".join(f"2026-05-01\tCompra {i}\t-{i}.50" for i in range(1, 251))
    content = f"{header}\n{rows}"

    resp = client.post(
        "/api/v1/parse",
        json={"user_id": "u1", "input_type": "text", "content": content, "filename": "x.csv"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert len(calls) > 1, "large statement should be chunked into multiple calls"
    assert body["metadata"]["count"] == len(calls)  # one merged row per chunk
    # The header is repeated on each batch so the model keeps the columns.
    assert all(header in c for c in calls)


def test_parse_survives_one_failed_chunk(client, monkeypatch):
    calls = {"n": 0}

    async def _fake(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise llm.LLMUnavailable("batch 2 down")
        return llm.LLMResult(
            data={"transactions": [{"date": "2026-05-01", "concept": "OK", "amount": -1.0}]},
            provider_used="nan_builders", is_fallback=False, primary_error=None,
        )

    monkeypatch.setattr(parse_router.llm, "complete_json", _fake)
    rows = "\n".join(f"2026-05-01\tCompra {i}\t-{i}.50" for i in range(1, 251))
    resp = client.post(
        "/api/v1/parse",
        json={"user_id": "u1", "input_type": "text", "content": rows, "filename": "x.csv"},
    )
    assert resp.status_code == 200  # other batches still import


# --- /parse/plan fast-lane gate (#speed cost guardrails) -----------------------

def _plan(client, chunk_count):
    return client.post("/api/v1/parse/plan", json={"user_id": "u1", "chunk_count": chunk_count})


def test_plan_grants_openai_for_a_large_import(client, monkeypatch):
    monkeypatch.setattr(parse_router.oplog, "fast_lane_count_this_month", lambda *_: 0)
    granted = {"n": 0}
    monkeypatch.setattr(parse_router.oplog, "record_fast_lane_grant",
                        lambda *_: granted.__setitem__("n", granted["n"] + 1))
    r = _plan(client, 10)  # >4 and <=50
    assert r.status_code == 200
    body = r.json()
    assert body["lane"] == "openai" and body["concurrency"] == 8
    assert granted["n"] == 1  # the grant was recorded (counts against the monthly quota)


def test_plan_small_import_stays_on_gemma(client, monkeypatch):
    granted = {"n": 0}
    monkeypatch.setattr(parse_router.oplog, "fast_lane_count_this_month", lambda *_: 0)
    monkeypatch.setattr(parse_router.oplog, "record_fast_lane_grant",
                        lambda *_: granted.__setitem__("n", granted["n"] + 1))
    r = _plan(client, 3)  # <= threshold
    assert r.json()["lane"] == "gemma"
    assert granted["n"] == 0  # no grant, no quota consumed


def test_plan_huge_import_stays_on_gemma(client, monkeypatch):
    monkeypatch.setattr(parse_router.oplog, "fast_lane_count_this_month", lambda *_: 0)
    monkeypatch.setattr(parse_router.oplog, "record_fast_lane_grant", lambda *_: None)
    r = _plan(client, 51)  # > max_chunks: too big, protect the bill
    assert r.json()["lane"] == "gemma"


def test_plan_denies_over_monthly_limit(client, monkeypatch):
    monkeypatch.setattr(parse_router.oplog, "fast_lane_count_this_month", lambda *_: 15)  # at the cap
    granted = {"n": 0}
    monkeypatch.setattr(parse_router.oplog, "record_fast_lane_grant",
                        lambda *_: granted.__setitem__("n", granted["n"] + 1))
    r = _plan(client, 10)
    assert r.json()["lane"] == "gemma"
    assert granted["n"] == 0


def test_parse_large_import_flag_routes_to_fallback(client, monkeypatch):
    seen = {}

    async def _capture(*args, **kwargs):
        seen["prefer_fallback"] = kwargs.get("prefer_fallback")
        return llm.LLMResult(
            data={"transactions": [{"date": "2026-05-01", "concept": "OK", "amount": -1.0}]},
            provider_used="openai", is_fallback=True, primary_error=None,
        )

    monkeypatch.setattr(parse_router.llm, "complete_json", _capture)
    resp = client.post("/api/v1/parse", json={
        "user_id": "u1", "input_type": "text", "content": "2026-05-01\tX\t-1.00",
        "filename": "x.csv", "large_import": True,
    })
    assert resp.status_code == 200
    assert seen["prefer_fallback"] is True
