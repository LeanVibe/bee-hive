from types import SimpleNamespace
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.main import create_app


def test_debug_agents_returns_error_payload_on_exception(monkeypatch):
    # No-op security middleware; permissive settings for isolated test
    import app.core.enterprise_security_system as ess
    import app.main as app_main
    import app.core.agent_spawner as spawner

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

    # Force get_active_agents_status to raise
    async def _boom():  # pragma: no cover
        raise RuntimeError("spawn failure")

    monkeypatch.setattr(spawner, "get_active_agents_status", lambda: _boom(), raising=True)

    app: FastAPI = create_app()
    client = TestClient(app, raise_server_exceptions=False)

    resp = client.get("/debug-agents")
    assert resp.status_code == 200
    body = resp.json()
    assert body.get("status") == "debug_error"
    assert "error" in body