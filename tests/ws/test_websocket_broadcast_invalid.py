import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.asyncio


async def test_ws_broadcast_invalid_subscription_returns_error(test_app):
    client = TestClient(test_app)

    resp = client.post(
        "/api/dashboard/websocket/broadcast",
        params={"subscription": "does-not-exist", "message_type": "noop"},
        json={"hello": "world"},
        headers={"host": "localhost:8000"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "error" in body
    assert body.get("valid_subscriptions")
