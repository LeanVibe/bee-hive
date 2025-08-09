import json
import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.asyncio


async def test_ws_unsubscribe_unknown_is_tolerated(test_app):
    client = TestClient(test_app)

    with client.websocket_connect("/api/dashboard/ws/dashboard", headers={"host": "localhost:8000"}) as ws:
        # Drain initial message
        _ = json.loads(ws.receive_text())
        # Unsubscribe from an unknown subscription should not crash; server should echo updated set
        ws.send_text(json.dumps({"type": "unsubscribe", "subscriptions": ["does-not-exist"]}))
        # Read a few frames to allow for interleaved periodic updates
        got_confirmation = False
        for _ in range(10):
            msg = json.loads(ws.receive_text())
            if msg.get("type") == "subscription_updated":
                got_confirmation = True
                break
        assert got_confirmation
