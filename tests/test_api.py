"""Tests for API endpoints"""

import pytest


class TestAPI:
    """Test suite for API endpoints."""

    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/api/health")
        assert response.status_code == 200
        assert response.json["status"] == "healthy"

    def test_predict_endpoint(self, client):
        """Test prediction endpoint."""
        response = client.post(
            "/api/predict",
            json={"input": "test_data"},
            content_type="application/json",
        )
        assert response.status_code == 200
        assert "result" in response.json

    def test_get_tasks(self, client):
        """Test get tasks endpoint."""
        response = client.get("/api/tasks")
        assert response.status_code == 200
        assert "tasks" in response.json


@pytest.fixture
def client():
    """Create test client."""
    from api.app import app

    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client
