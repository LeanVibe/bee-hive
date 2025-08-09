import asyncio
import json
import pytest

from app.api.dashboard_websockets import DashboardWebSocketManager, WebSocketConnection


class FailingWS:
    async def send_text(self, text):
        raise RuntimeError("simulated send failure")


@pytest.mark.asyncio
async def test_backpressure_disconnects_after_consecutive_failures(monkeypatch):
    mgr = DashboardWebSocketManager()
    mgr.backpressure_disconnect_threshold = 3

    conn_id = "bp1"
    conn = WebSocketConnection(
        websocket=FailingWS(),
        connection_id=conn_id,
        client_type="test",
        subscriptions=set(["agents"]),
        connected_at=mgr.connections.get(conn_id).connected_at if False else __import__(
            "datetime"
        ).datetime.utcnow(),
        last_activity=__import__("datetime").datetime.utcnow(),
        metadata={},
        tokens=10.0,
        last_refill=__import__("datetime").datetime.utcnow(),
        rate_limit_notified_at=None,
    )
    mgr.connections[conn_id] = conn
    mgr.subscription_groups["agents"].add(conn_id)

    # Broadcast to 'agents' will attempt to send and fail; after 3 failures
    # the connection should be disconnected and counter incremented
    await mgr.broadcast_to_subscription("agents", "agent_update", {"x": 1})
    await mgr.broadcast_to_subscription("agents", "agent_update", {"x": 2})
    await mgr.broadcast_to_subscription("agents", "agent_update", {"x": 3})

    assert conn_id not in mgr.connections
    assert mgr.metrics["backpressure_disconnects_total"] >= 1
