import json
import os
import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.asyncio


async def test_ws_auth_required_rejects_without_token(monkeypatch, test_app):
    # Enable auth required
    monkeypatch.setenv("WS_AUTH_REQUIRED", "true")
    monkeypatch.setenv("WS_AUTH_TOKEN", "secret-token")

    client = TestClient(test_app)

    # No Authorization header → server should not accept
    from starlette.websockets import WebSocketDisconnect
    try:
        with client.websocket_connect("/api/dashboard/ws/dashboard", headers={"host": "localhost:8000"}) as ws:
            # If context opens, receiving should fail immediately
            with pytest.raises(Exception):
                _ = ws.receive_text()
    except WebSocketDisconnect:
        pass


async def test_ws_auth_required_accepts_with_token(monkeypatch, test_app):
    monkeypatch.setenv("WS_AUTH_REQUIRED", "true")
    monkeypatch.setenv("WS_AUTH_TOKEN", "secret-token")

    client = TestClient(test_app)

    # Provide correct Authorization header
    with client.websocket_connect(
        "/api/dashboard/ws/dashboard",
        headers={
            "host": "localhost:8000",
            "Authorization": "Bearer secret-token",
        },
    ) as ws:
        payload = json.loads(ws.receive_text())
        assert payload.get("type") == "connection_established"


async def test_ws_allowlist_blocks_unknown_origin(monkeypatch, test_app):
    monkeypatch.setenv("WS_ALLOWED_ORIGINS", "https://good.example")

    client = TestClient(test_app)

    # Origin not in allowlist → server should not accept
    from starlette.websockets import WebSocketDisconnect
    try:
        with client.websocket_connect(
            "/api/dashboard/ws/dashboard",
            headers={
                "host": "localhost:8000",
                "Origin": "https://bad.example",
            },
        ) as ws:
            with pytest.raises(Exception):
                _ = ws.receive_text()
    except WebSocketDisconnect:
        pass


def test_ws_limits_exposes_auth_flags(test_app):
    client = TestClient(test_app)
    data = client.get("/api/dashboard/websocket/limits", headers={"host": "localhost:8000"}).json()
    assert "ws_auth_required" in data
    assert "ws_allowed_origins_configured" in data
    assert "idle_disconnect_seconds" in data
