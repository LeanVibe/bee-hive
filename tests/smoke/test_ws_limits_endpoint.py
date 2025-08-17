import pytest
from fastapi.testclient import TestClient


def test_ws_limits_endpoint(test_app):
    client = TestClient(test_app)
    resp = client.get("/api/dashboard/websocket/limits", headers={"host": "localhost:8000"})
    assert resp.status_code == 200
    data = resp.json()
    assert "rate_limit_tokens_per_second" in data
    assert "max_inbound_message_bytes" in data
    assert "max_subscriptions_per_connection" in data
    assert "backpressure_disconnect_threshold" in data
    assert "contract_version" in data
