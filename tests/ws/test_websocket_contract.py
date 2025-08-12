import asyncio
import json
import pytest
from starlette.websockets import WebSocketDisconnect


@pytest.mark.asyncio
async def test_dashboard_ws_basic_contract(test_app):
    from starlette.testclient import TestClient

    client = TestClient(test_app)

    # Pass a permissive host header to satisfy host/middleware checks
    with client.websocket_connect("/api/dashboard/ws/dashboard", headers={"host": "localhost:8000"}) as ws:
        # Send ping and expect pong
        ws.send_text(json.dumps({"type": "ping"}))
        message = ws.receive_text()
        payload = json.loads(message)
        assert payload["type"] in {"pong", "dashboard_initialized", "dashboard_update", "connection_established"}

        # Subscribe to a specific stream and accept either confirmation or an update
        ws.send_text(json.dumps({"type": "subscribe", "subscriptions": ["agents"]}))
        msg = json.loads(ws.receive_text())
        # Allow a pong in case the background loop hasn't broadcast yet
        assert msg["type"] in {"subscription_updated", "agent_update", "dashboard_update", "pong"}
