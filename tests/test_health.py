def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["database"] == "disabled"  # no DATABASE_URL in tests


def test_health_under_prefix(client):
    assert client.get("/api/v1/health").status_code == 200
