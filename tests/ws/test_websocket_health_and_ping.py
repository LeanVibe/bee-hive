import json
import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.asyncio


async def test_ws_health_endpoint_and_ping_pong_timestamp(test_app):
    client = TestClient(test_app)

    # Health endpoint should return basic structure
    r = client.get("/api/dashboard/websocket/health", headers={"host": "localhost:8000"})
    assert r.status_code == 200
    body = r.json()
    assert "websocket_manager" in body
    assert "background_tasks" in body
    assert "overall_health" in body

    # WS ping/pong should include a timestamp
    with client.websocket_connect("/api/dashboard/ws/dashboard", headers={"host": "localhost:8000"}) as ws:
        # Drain initial
        _ = json.loads(ws.receive_text())
        ws.send_text(json.dumps({"type": "ping"}))
        pong = None
        for _ in range(6):
            msg = json.loads(ws.receive_text())
            if msg.get("type") == "pong":
                pong = msg
                break
        assert pong is not None
        assert isinstance(pong.get("timestamp"), str)
