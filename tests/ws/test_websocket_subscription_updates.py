import asyncio
import json
import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.asyncio


def _read_json(ws):
    return json.loads(ws.receive_text())


async def test_ws_subscription_update_flow(test_app):
    client = TestClient(test_app)

    # Connect with default subscriptions
    ws = client.websocket_connect("/api/dashboard/ws/dashboard", headers={"host": "localhost:8000"})
    with ws:
        msg = _read_json(ws)
        assert msg["type"] in {"connection_established", "dashboard_initialized"}
        # Request to subscribe to 'alerts' (with duplicate + unknown to test validation and sorting)
        ws.send_text(json.dumps({"type": "subscribe", "subscriptions": ["alerts", "alerts", "unknown"]}))
        # Background loop may interleave updates; read a few messages until we see confirmation
        seen_update = False
        for _ in range(20):
            msg = _read_json(ws)
            if msg.get("type") == "subscription_updated":
                subs = msg["subscriptions"]
                assert subs == sorted(subs)
                assert "alerts" in subs
                seen_update = True
                break
            if msg.get("type") == "error" and "Invalid subscription" in msg.get("message", ""):
                # validation error for unknown should be raised
                pass
        assert seen_update, "Did not receive subscription_updated within expected messages"

        # Give the background loop a tick to possibly broadcast
        await asyncio.sleep(0.1)
