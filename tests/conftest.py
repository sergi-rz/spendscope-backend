"""Test fixtures. The suite never touches the network or a database:

  - DATABASE_URL is forced empty -> cache/rate-limit/oplog degrade to no-ops.
  - The LLM router and RevenueCat are monkeypatched per-test.
"""

from __future__ import annotations

import os

# Force a DB-less, premium-gated configuration BEFORE app modules import their settings.
os.environ.setdefault("DATABASE_URL", "")
os.environ["DATABASE_URL"] = ""
os.environ.setdefault("REQUIRE_PREMIUM", "true")

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)
