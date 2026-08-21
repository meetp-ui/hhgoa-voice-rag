from __future__ import annotations

from app.main import create_app


def test_health_degrades_gracefully_when_qdrant_unreachable() -> None:
    app = create_app()
    client = app.test_client()

    response = client.get("/health")

    assert response.status_code == 503
    data = response.get_json()
    assert data["status"] == "degraded"
    assert data["providers"]["qdrant"] == {"configured": True, "reachable": False}
    assert data["providers"]["sarvam"] == {"configured": True, "reachable": None}
    assert data["providers"]["ollama_cloud"] == {"configured": True, "reachable": None}
