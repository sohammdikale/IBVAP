"""Phase 1 tests: app boots and health endpoint responds correctly."""

from fastapi.testclient import TestClient

from backend.main import app

# Context-manager entry so FastAPI's lifespan (init_db) actually runs.
client = TestClient(app)
client.__enter__()


def test_health_endpoint_returns_200():
    response = client.get("/api/health")
    assert response.status_code == 200


def test_health_endpoint_reports_online():
    response = client.get("/api/health")
    body = response.json()
    assert body["status"] == "online"
    assert body["database"] == "connected"
    assert body["app_name"] == "IBVAP"
