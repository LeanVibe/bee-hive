import pytest
from datetime import datetime, timedelta

from app.api.dashboard_websockets import DashboardWebSocketManager, WebSocketConnection


@pytest.mark.asyncio
async def test_idle_disconnect_sends_notice_and_disconnects(monkeypatch):
    mgr = DashboardWebSocketManager()

    sent = []

    async def fake_send(connection_id, message):
        sent.append((connection_id, message))
        return True

    monkeypatch.setattr(mgr, "_send_to_connection", fake_send)

    class DummyWS:
        async def send_text(self, text):
            pass

    now = datetime.utcnow()
    conn = WebSocketConnection(
        websocket=DummyWS(),
        connection_id="idle-1",
        client_type="test",
        subscriptions=set(),
        connected_at=now - timedelta(minutes=20),
        last_activity=now - timedelta(minutes=20),
        metadata={},
        tokens=1.0,
        last_refill=now,
        rate_limit_notified_at=None,
    )

    mgr.connections["idle-1"] = conn
    mgr.idle_disconnect_seconds = 60  # 1 minute for test

    await mgr._check_idle_disconnects(now)

    # Should have sent a disconnect_notice and removed the connection
    assert any(m[1].get("type") == "disconnect_notice" and m[1].get("reason") == "idle_timeout" for m in sent)
    assert "idle-1" not in mgr.connections
