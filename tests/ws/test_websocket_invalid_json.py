import json
import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.asyncio


async def test_ws_invalid_json_message_returns_error(test_app):
    client = TestClient(test_app)

    with client.websocket_connect("/api/dashboard/ws/dashboard", headers={"host": "localhost:8000"}) as ws:
        # Drain initial message(s) that may include initialization or periodic updates
        _ = json.loads(ws.receive_text())
        # Send invalid JSON
        ws.send_text("not-json")
        # Read a few frames to find the error despite interleaved periodic messages
        found_error = False
        for _ in range(6):
            msg = json.loads(ws.receive_text())
            if msg.get("type") == "error" and "Invalid JSON" in msg.get("message", ""):
                found_error = True
                break
        assert found_error
