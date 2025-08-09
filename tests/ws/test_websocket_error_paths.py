import json
import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.asyncio


async def test_ws_unknown_message_type_returns_error(test_app):
    client = TestClient(test_app)

    with client.websocket_connect("/api/dashboard/ws/dashboard", headers={"host": "localhost:8000"}) as ws:
        # Drain initial
        _ = json.loads(ws.receive_text())
        # Send unknown message type
        ws.send_text(json.dumps({"type": "unknown_thing", "foo": "bar"}))
        # Read several frames to find error amidst interleaved updates
        got_error = False
        error_msg = None
        for _ in range(8):
            msg = json.loads(ws.receive_text())
            if msg.get("type") == "error" and "Unknown message type" in msg.get("message", ""):
                got_error = True
                error_msg = msg
                break
        assert got_error
        # Error frames must include timestamp for observability
        assert isinstance(error_msg.get("timestamp"), str)


async def test_ws_request_data_unknown_type_produces_data_error(test_app):
    client = TestClient(test_app)

    with client.websocket_connect("/api/dashboard/ws/dashboard", headers={"host": "localhost:8000"}) as ws:
        # Drain initial
        _ = json.loads(ws.receive_text())
        # Send unknown data request
        ws.send_text(json.dumps({"type": "request_data", "data_type": "does_not_exist"}))
        got_data_error = None
        for _ in range(10):
            msg = json.loads(ws.receive_text())
            if msg.get("type") == "data_error" and msg.get("data_type") == "does_not_exist":
                got_data_error = msg
                break
        assert got_data_error is not None
        # Should include timestamp for observability
        assert isinstance(got_data_error.get("timestamp"), str)
        # Ensure error message present
        assert isinstance(got_data_error.get("error"), str) and got_data_error["error"]
