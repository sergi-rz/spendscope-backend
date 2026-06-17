import base64

from app.routers import extract as extract_router


def test_extract_text_passthrough(client):
    # Plain text needs no flattening — it comes back verbatim as kind="text".
    resp = client.post(
        "/api/v1/extract",
        json={"user_id": "u1", "input_type": "text", "content": "Fecha\tConcepto\t-12,50"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["kind"] == "text"
    assert body["text"] == "Fecha\tConcepto\t-12,50"


def test_extract_excel_returns_flattened_text(client, monkeypatch):
    # A binary spreadsheet is flattened to rows (openpyxl) and returned as text so the app can chunk it.
    monkeypatch.setattr(extract_router, "prepare", lambda *a, **k: ("text", "Fecha\tImporte\n2026-05-01\t-9.99"))
    fake_xlsx = base64.b64encode(b"PK\x03\x04fake").decode()
    resp = client.post(
        "/api/v1/extract",
        json={"user_id": "u1", "input_type": "image", "content": fake_xlsx, "filename": "x.xlsx"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["kind"] == "text"
    assert "2026-05-01" in body["text"]


def test_extract_real_image_reports_kind_image(client):
    # A genuine photo can't be flattened → kind="image", text=null; the app sends it straight to /parse.
    png = base64.b64encode(b"\x89PNG\r\n\x1a\n\x00\x00\x00\x00").decode()
    resp = client.post(
        "/api/v1/extract",
        json={"user_id": "u1", "input_type": "image", "content": png, "filename": "photo.png"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["kind"] == "image"
    assert body["text"] is None


def test_extract_invalid_base64_returns_400(client):
    resp = client.post(
        "/api/v1/extract",
        json={"user_id": "u1", "input_type": "image", "content": "not base64!!", "filename": "x.pdf"},
    )
    assert resp.status_code == 400


def test_extract_requires_user_id(client):
    resp = client.post("/api/v1/extract", json={"input_type": "text", "content": "x"})
    assert resp.status_code == 422
