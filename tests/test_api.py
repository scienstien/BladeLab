"""Tests for API endpoints"""

import pytest
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def client():
    """Create test client."""
    from api.app import app

    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def valid_observation():
    return {
        "efficiency": 0.65,
        "pressure_ratio": 2.3,
        "mass_flow": 0.9,
        "feasible": True,
        "surge_margin": 0.15,
        "choke_margin": 0.1,
        "r2": 0.08,
        "blade_angle": 60.0,
        "b2": 0.012,
        "Z": 7,
    }


def assert_validation_error(response, *expected_fragments):
    assert response.status_code == 400
    assert response.json["error"] == "Validation failed"
    assert isinstance(response.json["details"], list)
    for fragment in expected_fragments:
        assert any(fragment.lower() in detail.lower() for detail in response.json["details"])


class TestAPI:
    """Test suite for API endpoints."""

    def test_health_endpoint(self, client):
        """Test health check endpoint returns healthy status."""
        response = client.get("/api/health")
        assert response.status_code == 200
        assert response.json["status"] == "healthy"

    def test_tasks_endpoint(self, client):
        """Test tasks endpoint returns 3 tasks."""
        response = client.get("/api/tasks")
        assert response.status_code == 200
        assert "tasks" in response.json
        tasks = response.json["tasks"]
        assert len(tasks) == 3
        assert "feasibility" in tasks
        assert "target_pr" in tasks
        assert "target_pr_efficiency" in tasks

    def test_predict_with_heuristic_policy(self, client, valid_observation):
        """Test predict endpoint with valid observation and heuristic policy."""
        response = client.post(
            "/api/predict",
            json={
                "observation": valid_observation,
                "task_name": "feasibility",
                "policy_type": "heuristic",
            },
            content_type="application/json",
        )
        assert response.status_code == 200
        assert "action" in response.json
        assert response.json["policy_type"] == "heuristic"
        assert response.json["observation"] == valid_observation
        assert "delta_r2" in response.json["action"]

    def test_predict_missing_observation(self, client):
        """Test predict endpoint without observation field returns 400."""
        response = client.post(
            "/api/predict",
            json={"policy_type": "heuristic"},
            content_type="application/json",
        )
        assert_validation_error(response, "observation")

    def test_predict_invalid_policy_type(self, client, valid_observation):
        """Test predict endpoint with invalid policy_type returns 400."""
        response = client.post(
            "/api/predict",
            json={
                "observation": valid_observation,
                "policy_type": "invalid_policy",
            },
            content_type="application/json",
        )
        assert_validation_error(response, "policy_type")

    def test_predict_openai_requires_model_name(self, client, valid_observation):
        """Test predict endpoint requires model_name for openai policy."""
        response = client.post(
            "/api/predict",
            json={
                "observation": valid_observation,
                "task_name": "feasibility",
                "policy_type": "openai",
            },
            content_type="application/json",
        )
        assert_validation_error(response, "model_name")

    def test_rollout_endpoint(self, client):
        """Test rollout endpoint with task_name and policy_type."""
        response = client.post(
            "/api/rollout",
            json={
                "task_name": "feasibility",
                "policy_type": "heuristic",
                "max_steps": 50,
            },
            content_type="application/json",
        )
        assert response.status_code == 200
        assert "trajectory" in response.json
        assert "total_reward" in response.json
        assert "success" in response.json
        assert "steps" in response.json
        assert "scores" in response.json

    def test_rollout_openai_requires_model_name(self, client):
        """Test rollout endpoint requires model_name for openai policy."""
        response = client.post(
            "/api/rollout",
            json={
                "task_name": "feasibility",
                "policy_type": "openai",
            },
            content_type="application/json",
        )
        assert_validation_error(response, "model_name")

    def test_evaluate_endpoint(self, client):
        """Test evaluate endpoint with num_episodes."""
        response = client.post(
            "/api/evaluate",
            json={
                "num_episodes": 5,
                "task_name": "target_pr",
                "policy_type": "heuristic",
            },
            content_type="application/json",
        )
        assert response.status_code == 200
        assert response.json["task_name"] == "target_pr"
        assert response.json["policy_type"] == "heuristic"
        assert "num_episodes" in response.json
        assert response.json["num_episodes"] == 5
        assert "reward_mean" in response.json
        assert "reward_variance" in response.json
        assert "pr_mean" in response.json
        assert "pr_variance" in response.json
        assert "efficiency_mean" in response.json
        assert "efficiency_variance" in response.json
        assert "mass_flow_mean" in response.json
        assert "mass_flow_variance" in response.json

    def test_evaluate_missing_task_name(self, client):
        """Test evaluate endpoint requires task_name."""
        response = client.post(
            "/api/evaluate",
            json={"policy_type": "heuristic"},
            content_type="application/json",
        )
        assert_validation_error(response, "task_name")

    def test_predict_with_openai_policy(self, client, monkeypatch, valid_observation):
        """Test predict endpoint with openai policy type."""
        from inference import HeuristicPolicy

        captured = {}

        def fake_load_openai_policy(task_name, model_name):
            captured["task_name"] = task_name
            captured["model_name"] = model_name
            return HeuristicPolicy()

        monkeypatch.setattr("api.routes.load_openai_policy", fake_load_openai_policy)

        response = client.post(
            "/api/predict",
            json={
                "observation": valid_observation,
                "task_name": "feasibility",
                "policy_type": "openai",
                "model_name": "gpt-4.1-mini",
            },
            content_type="application/json",
        )
        assert response.status_code == 200
        assert "action" in response.json
        assert response.json["policy_type"] == "openai"
        assert response.json["observation"] == valid_observation
        assert captured == {"task_name": "feasibility", "model_name": "gpt-4.1-mini"}

    def test_predict_default_policy(self, client, valid_observation):
        """Test predict endpoint uses default heuristic policy when not specified."""
        response = client.post(
            "/api/predict",
            json={"observation": valid_observation, "task_name": "feasibility"},
            content_type="application/json",
        )
        assert response.status_code == 200
        assert "action" in response.json
        assert response.json["policy_type"] == "heuristic"
        assert response.json["observation"] == valid_observation

    def test_rollout_missing_task_name(self, client):
        """Test rollout endpoint without task_name returns 400."""
        response = client.post(
            "/api/rollout",
            json={"policy_type": "heuristic"},
            content_type="application/json",
        )
        assert_validation_error(response, "task_name")

    def test_rollout_missing_policy_type(self, client):
        """Test rollout endpoint without policy_type returns 400."""
        response = client.post(
            "/api/rollout",
            json={"task_name": "feasibility"},
            content_type="application/json",
        )
        assert_validation_error(response, "policy_type")

    def test_rollout_invalid_task_name(self, client):
        """Test rollout endpoint with invalid task_name returns 400."""
        response = client.post(
            "/api/rollout",
            json={
                "task_name": "invalid_task",
                "policy_type": "heuristic",
            },
            content_type="application/json",
        )
        assert_validation_error(response, "task_name")

    def test_rollout_invalid_policy_type(self, client):
        """Test rollout endpoint with invalid policy_type returns 400."""
        response = client.post(
            "/api/rollout",
            json={
                "task_name": "feasibility",
                "policy_type": "invalid_policy",
            },
            content_type="application/json",
        )
        assert_validation_error(response, "policy_type")

    def test_evaluate_defaults_num_episodes(self, client):
        """Test evaluate endpoint defaults num_episodes when omitted."""
        response = client.post(
            "/api/evaluate",
            json={"task_name": "feasibility"},
            content_type="application/json",
        )
        assert response.status_code == 200
        assert response.json["task_name"] == "feasibility"
        assert response.json["policy_type"] == "heuristic"
        assert response.json["num_episodes"] == 10

    def test_evaluate_invalid_num_episodes(self, client):
        """Test evaluate endpoint with invalid num_episodes returns 400."""
        response = client.post(
            "/api/evaluate",
            json={"task_name": "feasibility", "num_episodes": 0},
            content_type="application/json",
        )
        assert_validation_error(response, "num_episodes")

    def test_evaluate_negative_num_episodes(self, client):
        """Test evaluate endpoint with negative num_episodes returns 400."""
        response = client.post(
            "/api/evaluate",
            json={"task_name": "feasibility", "num_episodes": -5},
            content_type="application/json",
        )
        assert_validation_error(response, "num_episodes")

    def test_evaluate_invalid_task_name(self, client):
        """Test evaluate endpoint with invalid task_name returns 400."""
        response = client.post(
            "/api/evaluate",
            json={
                "num_episodes": 5,
                "task_name": "invalid_task",
            },
            content_type="application/json",
        )
        assert_validation_error(response, "task_name")

    def test_evaluate_invalid_policy_type(self, client):
        """Test evaluate endpoint with invalid policy_type returns 400."""
        response = client.post(
            "/api/evaluate",
            json={
                "num_episodes": 5,
                "task_name": "feasibility",
                "policy_type": "invalid_policy",
            },
            content_type="application/json",
        )
        assert_validation_error(response, "policy_type")

    def test_predict_empty_json(self, client):
        """Test predict endpoint with empty JSON returns 400."""
        response = client.post(
            "/api/predict",
            json={},
            content_type="application/json",
        )
        assert_validation_error(response, "observation")

    def test_rollout_empty_json(self, client):
        """Test rollout endpoint with empty JSON returns 400."""
        response = client.post(
            "/api/rollout",
            json={},
            content_type="application/json",
        )
        assert_validation_error(response, "task_name", "policy_type")
