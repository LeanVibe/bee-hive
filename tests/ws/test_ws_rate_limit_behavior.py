import json
import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.asyncio


async def test_ws_rate_limit_behavior_with_cooldown(test_app):
    client = TestClient(test_app)

    with client.websocket_connect("/api/dashboard/ws/dashboard", headers={"host": "localhost:8000"}) as ws:
        # Drain initial connection_established
        _ = json.loads(ws.receive_text())

        # Rapidly send > burst messages to trigger rate limiting
        # We do not await responses for each send; the server processes sequentially
        for _ in range(50):
            ws.send_text(json.dumps({"type": "ping"}))

        # Collect frames and ensure at least one rate limit error is observed.
        # We read enough frames to pass the burst (40) and observe the first limited response.
        got_rate_limit_error = False
        errors_seen = 0
        messages_seen = 0
        for _ in range(80):
            try:
                msg = json.loads(ws.receive_text())
            except Exception:
                break
            messages_seen += 1
            if msg.get("type") == "error" and "Rate limit" in (msg.get("message") or ""):
                got_rate_limit_error = True
                errors_seen += 1
        assert got_rate_limit_error, "Expected at least one rate limit error frame"
        # Cooldown should suppress a flood of identical errors
        assert errors_seen <= 5
