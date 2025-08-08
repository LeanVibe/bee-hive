import json
import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.asyncio


async def test_ws_broadcast_to_agents(test_app):
    client = TestClient(test_app)

    # Open two dashboard connections (default includes 'agents' subscription)
    ws1 = client.websocket_connect("/api/dashboard/ws/dashboard", headers={"host": "localhost:8000"})
    ws2 = client.websocket_connect("/api/dashboard/ws/dashboard", headers={"host": "localhost:8000"})

    with ws1, ws2:
        # Drain initial messages
        _ = json.loads(ws1.receive_text())
        _ = json.loads(ws2.receive_text())

        # Broadcast a message to 'agents'
        resp = client.post(
            "/api/dashboard/websocket/broadcast",
            params={"subscription": "agents", "message_type": "agent_update"},
            json={"test": True},
            headers={"host": "localhost:8000"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("success") is True
        assert data.get("subscription") == "agents"
        assert data.get("clients_reached", 0) >= 1
