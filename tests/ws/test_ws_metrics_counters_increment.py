import json
import re
import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.asyncio


async def test_ws_metrics_counters_increment_on_broadcast(test_app):
    client = TestClient(test_app)

    # Open a dashboard connection and subscribe to 'agents'
    with client.websocket_connect("/api/dashboard/ws/dashboard", headers={"host": "localhost:8000"}) as ws:
        _ = json.loads(ws.receive_text())
        ws.send_text(json.dumps({"type": "subscribe", "subscriptions": ["agents"]}))
        _ = json.loads(ws.receive_text())

        # Snapshot metrics before
        before_txt = client.get("/api/dashboard/metrics/websockets", headers={"host": "localhost:8000"}).text

        def get_counter(body: str, name: str) -> int:
            m = re.search(rf"^{re.escape(name)}\s+(\d+)$", body, re.MULTILINE)
            return int(m.group(1)) if m else 0

        before_sent = get_counter(before_txt, "leanvibe_ws_messages_sent_total")

        # Broadcast to agents
        res = client.post(
            "/api/dashboard/websocket/broadcast",
            params={"subscription": "agents", "message_type": "agent_update"},
            headers={"host": "localhost:8000"},
            json={"active_count": 1},
        )
        assert res.status_code == 200

        # Consume broadcast frame to avoid backlog
        for _ in range(5):
            msg = json.loads(ws.receive_text())
            if msg.get("type") == "agent_update":
                break

        # Metrics after
        after_txt = client.get("/api/dashboard/metrics/websockets", headers={"host": "localhost:8000"}).text
        after_sent = get_counter(after_txt, "leanvibe_ws_messages_sent_total")

        assert after_sent > before_sent
