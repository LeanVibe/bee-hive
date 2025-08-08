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
