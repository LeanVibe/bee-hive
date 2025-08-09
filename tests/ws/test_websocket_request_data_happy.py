import json
import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.asyncio


async def test_request_data_agent_status_happy_path(test_app):
    client = TestClient(test_app)

    with client.websocket_connect("/api/dashboard/ws/dashboard", headers={"host": "localhost:8000"}) as ws:
        # Drain initial
        _ = json.loads(ws.receive_text())

        ws.send_text(json.dumps({"type": "request_data", "data_type": "agent_status"}))

        got = None
        # Interleaving is possible; scan a few messages
        for _ in range(8):
            msg = json.loads(ws.receive_text())
            if msg.get("type") == "data_response" and msg.get("data_type") == "agent_status":
                got = msg
                break

        assert got is not None
        assert isinstance(got.get("data"), dict)
        assert "agents" in got["data"]
        assert "summary" in got["data"]


async def test_request_data_other_happy_paths(test_app):
    client = TestClient(test_app)

    with client.websocket_connect("/api/dashboard/ws/dashboard", headers={"host": "localhost:8000"}) as ws:
        _ = json.loads(ws.receive_text())

        # coordination_metrics
        ws.send_text(json.dumps({"type": "request_data", "data_type": "coordination_metrics"}))
        got_coord = None
        for _ in range(8):
            msg = json.loads(ws.receive_text())
            if msg.get("type") == "data_response" and msg.get("data_type") == "coordination_metrics":
                got_coord = msg
                break
        assert got_coord is not None
        assert isinstance(got_coord.get("data"), dict)
        assert "success_rate" in got_coord["data"]
        assert "trend" in got_coord["data"]

        # system_health
        ws.send_text(json.dumps({"type": "request_data", "data_type": "system_health"}))
        got_sys = None
        for _ in range(8):
            msg = json.loads(ws.receive_text())
            if msg.get("type") == "data_response" and msg.get("data_type") == "system_health":
                got_sys = msg
                break
        assert got_sys is not None
        assert isinstance(got_sys.get("data"), dict)
        assert "overall_status" in got_sys["data"]
        assert "components" in got_sys["data"]
