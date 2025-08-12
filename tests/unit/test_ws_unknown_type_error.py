import json
from types import SimpleNamespace
from fastapi.testclient import TestClient

from app.main import create_app


def test_ws_unknown_message_type_returns_error(monkeypatch):
    import app.core.enterprise_security_system as ess
    import app.main as app_main

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

    app = create_app()
    client = TestClient(app)

    with client.websocket_connect("/api/dashboard/ws/agents") as ws:
        # Drain initial frame(s)
        _first = ws.receive_text()

        ws.send_text(json.dumps({"type": "bogus"}))
        found_error = False
        for _ in range(5):
            msg = ws.receive_text()
            data = json.loads(msg)
            if data.get("type") == "error" and "Unknown message type" in data.get("message", ""):
                found_error = True
                break
        assert found_error, "Expected error for unknown message type"
