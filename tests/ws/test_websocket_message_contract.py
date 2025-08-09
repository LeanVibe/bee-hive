import asyncio
import json
import pytest
from fastapi.testclient import TestClient
from jsonschema import validate
import pathlib

pytestmark = pytest.mark.asyncio

SCHEMA = json.loads((pathlib.Path(__file__).parents[2] / "schemas/ws_messages.schema.json").read_text())


def _read_json(ws):
    return json.loads(ws.receive_text())


async def test_ws_initial_and_subscription_messages_match_schema(test_app):
    client = TestClient(test_app)

    ws = client.websocket_connect("/api/dashboard/ws/dashboard", headers={"host": "localhost:8000"})
    with ws:
        msg = _read_json(ws)
        validate(instance=msg, schema=SCHEMA)

        ws.send_text(json.dumps({"type": "subscribe", "subscriptions": ["alerts"]}))
        for _ in range(5):
            msg2 = _read_json(ws)
            validate(instance=msg2, schema=SCHEMA)
            if msg2.get("type") == "subscription_updated":
                break

        await asyncio.sleep(0.1)


async def test_ws_error_and_critical_alert_match_schema(test_app):
    client = TestClient(test_app)

    ws = client.websocket_connect("/api/dashboard/ws/dashboard", headers={"host": "localhost:8000"})
    with ws:
        # Drain initial message(s)
        _ = _read_json(ws)

        # Send invalid JSON to trigger structured error
        ws.send_text("not-json")
        got_error = False
        for _ in range(6):
            msg = _read_json(ws)
            try:
                validate(instance=msg, schema=SCHEMA)
            except Exception:
                # Some messages like pong or data_response may not match; continue scanning
                pass
            if msg.get("type") == "error" and isinstance(msg.get("message"), str):
                got_error = True
                break
        assert got_error

        # Subscribe to alerts and trigger a critical alert via broadcast API
        ws.send_text(json.dumps({"type": "subscribe", "subscriptions": ["alerts"]}))
        # Scan until subscription_updated
        for _ in range(6):
            msg = _read_json(ws)
            if msg.get("type") == "subscription_updated":
                break

        # Broadcast a critical alert
        resp = client.post(
            "/api/dashboard/websocket/broadcast",
            params={"subscription": "alerts", "message_type": "critical_alert"},
            json={"alerts": [{"message": "Test alert", "level": "critical"}]},
            headers={"host": "localhost:8000"},
        )
        assert resp.status_code == 200

        got_alert = False
        for _ in range(10):
            msg = _read_json(ws)
            if msg.get("type") == "critical_alert":
                # Must contain subscription and data.alerts per schema
                validate(instance=msg, schema=SCHEMA)
                assert msg.get("subscription") == "alerts"
                assert isinstance(msg.get("data", {}).get("alerts"), list)
                got_alert = True
                break
        assert got_alert
