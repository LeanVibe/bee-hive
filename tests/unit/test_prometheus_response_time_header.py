from types import SimpleNamespace
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.main import create_app


def test_prometheus_middleware_sets_response_time_header(monkeypatch):
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

    app: FastAPI = create_app()
    client = TestClient(app, raise_server_exceptions=False)

    resp = client.get("/status")
    assert resp.status_code == 200
    assert "X-Response-Time" in resp.headers
    assert resp.headers["X-Response-Time"].endswith("s")
