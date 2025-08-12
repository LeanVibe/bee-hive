import json
from types import SimpleNamespace
from fastapi.testclient import TestClient

from app.main import create_app


def test_ws_oversized_message_returns_too_large_error(monkeypatch):
    import app.core.enterprise_security_system as ess
    import app.main as app_main
    import app.api.dashboard_websockets as wsmod

    class _NoopSecurityMiddleware:
        async def __call__(self, request, call_next):
            return await call_next(request)

    monkeypatch.setattr(ess, "SecurityMiddleware", lambda: _NoopSecurityMiddleware())
    monkeypatch.setattr(
        app_main,
        "get_settings",
        lambda: SimpleNamespace(DEBUG=True, ALLOWED_HOSTS=["*"], CORS_ORIGINS=["*"]),
        raising=True,
    )

    # Force a tiny max to trigger the too large logic quickly
    monkeypatch.setattr(wsmod.websocket_manager, "max_inbound_message_bytes", 8, raising=True)

    app = create_app()
    client = TestClient(app)

    with client.websocket_connect("/api/dashboard/ws/agents") as ws:
        _first = ws.receive_text()  # drain welcome
        # Create a JSON message that will exceed 8 bytes once encoded
        payload = {"type": "ping", "pad": "xxxxxxxxxxxxx"}
        ws.send_text(json.dumps(payload))
        found_error = False
        for _ in range(5):
            msg = ws.receive_text()
            data = json.loads(msg)
            if data.get("type") == "error" and "too large" in data.get("message", "").lower():
                found_error = True
                break
        assert found_error, "Expected error for oversized message"
