import asyncio
import pytest
from datetime import datetime, timedelta

from app.api.dashboard_websockets import DashboardWebSocketManager, WebSocketConnection


@pytest.mark.asyncio
async def test_token_bucket_refill_and_consume(monkeypatch):
    mgr = DashboardWebSocketManager()

    class DummyWS:
        async def send_text(self, text):
            pass

    conn = WebSocketConnection(
        websocket=DummyWS(),
        connection_id="c1",
        client_type="test",
        subscriptions=set(),
        connected_at=datetime.utcnow(),
        last_activity=datetime.utcnow(),
        metadata={},
        tokens=1.0,
        last_refill=datetime.utcnow(),
        rate_limit_notified_at=None,
    )

    # consume one token
    assert mgr._consume_token_allow(conn) is True
    # now tokens likely below 1
    assert mgr._consume_token_allow(conn) is False

    # advance time to refill
    old_refill = conn.last_refill
    conn.last_refill = conn.last_refill - timedelta(seconds=1)
    assert mgr._consume_token_allow(conn) is True


@pytest.mark.asyncio
async def test_rate_limit_notifies_then_suppresses(monkeypatch):
    mgr = DashboardWebSocketManager()

    sent = []

    async def fake_send(connection_id, message):
        sent.append(message)
        return True

    monkeypatch.setattr(mgr, "_send_to_connection", fake_send)

    class DummyWS:
        async def accept(self):
            pass
        async def receive_text(self):
            return "{}"

    # create connection with zero tokens to trigger limit
    connection = WebSocketConnection(
        websocket=DummyWS(),
        connection_id="c2",
        client_type="test",
        subscriptions=set(),
        connected_at=datetime.utcnow(),
        last_activity=datetime.utcnow(),
        metadata={},
        tokens=0.0,
        last_refill=datetime.utcnow(),
        rate_limit_notified_at=None,
    )
    mgr.connections["c2"] = connection

    # first over-limit should notify
    await mgr.handle_message("c2", {"type": "ping"})
    assert any(m.get("type") == "error" and "Rate limit" in m.get("message", "") for m in sent)
    sent.clear()

    # immediate second over-limit should be suppressed due to cooldown
    await mgr.handle_message("c2", {"type": "ping"})
    assert len(sent) == 0
