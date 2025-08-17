import pytest
from fastapi.testclient import TestClient


def test_websocket_metrics_endpoint_exposes_core_counters(test_app):
    client = TestClient(test_app)

    resp = client.get("/api/dashboard/metrics/websockets", headers={"host": "localhost:8000"})
    assert resp.status_code == 200
    body = resp.text
    # Basic presence checks
    assert "leanvibe_websocket_connections_total" in body
    assert "leanvibe_websocket_connections_active" in body
    assert "leanvibe_websocket_subscriptions" in body
    # New counters from manager
    assert "leanvibe_ws_messages_sent_total" in body
    assert "leanvibe_ws_messages_send_failures_total" in body
    assert "leanvibe_ws_messages_received_total" in body
    assert "leanvibe_ws_messages_dropped_rate_limit_total" in body
    assert "leanvibe_ws_errors_sent_total" in body
    assert "leanvibe_ws_connections_total" in body
    assert "leanvibe_ws_disconnections_total" in body
    assert "leanvibe_ws_backpressure_disconnects_total" in body
