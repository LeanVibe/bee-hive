import json
from types import SimpleNamespace
from fastapi.testclient import TestClient

from app.main import create_app


def test_ws_invalid_json_yields_error_frame(monkeypatch):
    # No-op security; permissive settings
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
        # Drain initial welcome or broadcast frame(s)
        _first = ws.receive_text()

        ws.send_text("{invalid_json}")
        # Read a few frames and ensure an error appears
        found_error = False
        for _ in range(5):
            msg = ws.receive_text()
            data = json.loads(msg)
            if data.get("type") == "error" and "Invalid JSON" in data.get("message", ""):
                found_error = True
                break
        assert found_error, "Expected error frame for invalid JSON within 5 frames"
