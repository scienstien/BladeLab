"""Tests for the FastAPI Space application."""

from pathlib import Path
import sys

from fastapi.testclient import TestClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def test_space_root_returns_status_payload():
    from server.app import app

    client = TestClient(app)

    response = client.get("/")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "running"
    assert payload["name"] == "TurboDesigner 2.0"
    assert payload["docs"] == "/docs"
    assert payload["schema"] == "/schema"