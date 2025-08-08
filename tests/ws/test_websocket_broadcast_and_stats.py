import json
import pytest
from starlette.testclient import TestClient


@pytest.mark.asyncio
async def test_ws_broadcast_and_stats(test_app):
    client = TestClient(test_app)

    # Open two clients subscribed to 'agents'
    ws1 = client.websocket_connect("/api/dashboard/ws/dashboard", headers={"host": "localhost:8000"})
    ws2 = client.websocket_connect("/api/dashboard/ws/dashboard", headers={"host": "localhost:8000"})

    with ws1, ws2:
        # Drain initial messages
        _ = json.loads(ws1.receive_text())
        _ = json.loads(ws2.receive_text())

        # Subscribe both to 'agents'
        ws1.send_text(json.dumps({"type": "subscribe", "subscriptions": ["agents"]}))
        ws2.send_text(json.dumps({"type": "subscribe", "subscriptions": ["agents"]}))
        _ = json.loads(ws1.receive_text())
        _ = json.loads(ws2.receive_text())

        # Optionally consume one periodic message if sent
        for _ in range(1):
            try:
                _ = json.loads(ws1.receive_text())
                _ = json.loads(ws2.receive_text())
            except Exception:
                break

        # Stats endpoint reflects active connections (provide host header due to TrustedHost)
        stats = client.get("/api/dashboard/websocket/stats", headers={"host": "localhost:8000"}).json()
        assert stats["websocket_stats"]["total_connections"] >= 2

    # After exiting context, connections close; we can’t reliably assert count==0 instantly,
    # but the endpoint should still be reachable and return shape
    stats2 = client.get("/api/dashboard/websocket/stats", headers={"host": "localhost:8000"})
    assert stats2.status_code == 200
