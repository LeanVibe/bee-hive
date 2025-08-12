import json
import pytest


@pytest.mark.asyncio
async def test_dashboard_ws_initial_frame_is_connection_established(test_app):
    from starlette.testclient import TestClient

    client = TestClient(test_app)

    # Connect and immediately read first server frame
    with client.websocket_connect(
        "/api/dashboard/ws/dashboard", headers={"host": "localhost:8000"}
    ) as ws:
        message = ws.receive_text()
        payload = json.loads(message)

        assert payload["type"] == "connection_established"
        assert "connection_id" in payload
        assert "contract_version" in payload
