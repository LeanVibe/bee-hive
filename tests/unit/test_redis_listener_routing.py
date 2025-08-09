import asyncio
import json
import pytest
from app.api.dashboard_websockets import DashboardWebSocketManager


@pytest.mark.asyncio
async def test_handle_redis_event_routes_system_and_agent_messages(monkeypatch):
    mgr = DashboardWebSocketManager()

    # Pretend subscriptions exist so routing path is exercised
    mgr.subscription_groups["system"].add("conn1")
    mgr.subscription_groups["agents"].add("conn2")

    calls = []

    async def fake_broadcast(subscription, message_type, data):
        calls.append((subscription, message_type, data))
        return 1

    monkeypatch.setattr(mgr, "broadcast_to_subscription", fake_broadcast)

    # system message
    await mgr._handle_redis_event({
        "type": "message",
        "channel": b"system_events",
        "data": json.dumps({"ok": True}).encode(),
    })

    # pattern agent message
    await mgr._handle_redis_event({
        "type": "pmessage",
        "pattern": b"agent_events:*",
        "channel": b"agent_events:alpha",
        "data": json.dumps({"agent": "alpha"}).encode(),
    })

    assert ("system", "system_event", {"ok": True}) in calls
    assert ("agents", "agent_event", {"agent": "alpha"}) in calls
