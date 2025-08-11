import importlib
import os
from types import SimpleNamespace
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.main import create_app


def test_global_exception_handler_returns_500_and_payload(monkeypatch):
    # Patch enterprise security middleware to a no-op for unit test
    import app.core.enterprise_security_system as ess
    import app.main as app_main

    class _NoopSecurityMiddleware:
        async def __call__(self, request, call_next):
            return await call_next(request)

    monkeypatch.setattr(ess, "SecurityMiddleware", lambda: _NoopSecurityMiddleware())

    # Allow all hosts / origins to avoid 400 from TrustedHost/CORS in tests
    monkeypatch.setattr(
        app_main,
        "get_settings",
        lambda: SimpleNamespace(DEBUG=True, ALLOWED_HOSTS=["*"], CORS_ORIGINS=["*"]),
        raising=True,
    )

    app: FastAPI = create_app()

    @app.get("/boom")
    async def boom():  # type: ignore[no-redef]
        raise RuntimeError("kaboom")

    # Ensure server exceptions are returned as 500 responses, not re-raised
    client = TestClient(app, raise_server_exceptions=False)
    resp = client.get("/boom")
    assert resp.status_code == 500
    data = resp.json()
    assert data.get("error") == "Internal server error"
    assert "request_id" in data


def test_ci_minimal_app_health_endpoint(monkeypatch):
    # Ensure re-import picks CI mode
    monkeypatch.setenv("CI", "true")
    # Reload module to apply CI flag
    import app.main as app_main

    importlib.reload(app_main)
    assert hasattr(app_main, "app")

    client = TestClient(app_main.app)  # type: ignore[arg-type]
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json().get("ci") is True
