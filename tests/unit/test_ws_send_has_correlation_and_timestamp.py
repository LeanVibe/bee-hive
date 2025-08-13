import json
from starlette.testclient import TestClient

from app.main import create_app


def test_ws_outbound_injects_correlation_id_and_timestamp(monkeypatch):
    # Create app and connect
    app = create_app()
    client = TestClient(app)

    with client.websocket_connect("/api/dashboard/ws/dashboard", headers={"host": "localhost:8000"}) as ws:
        # First frame from server should include correlation_id and timestamp
        initial = json.loads(ws.receive_text())
        assert "correlation_id" in initial
        assert "timestamp" in initial or initial.get("type") == "connection_established"

        # Ask for a data request that triggers a server response
        ws.send_text(json.dumps({"type": "request_data", "data_type": "agent_status"}))
        for _ in range(5):
            # tolerate non-JSON frames if any
            try:
                msg = json.loads(ws.receive_text())
            except json.JSONDecodeError:
                continue
            if msg.get("type") in {"data_response", "error", "data_error"}:
                assert "correlation_id" in msg
                assert "timestamp" in msg
                break
