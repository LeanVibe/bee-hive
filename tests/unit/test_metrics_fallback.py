from types import SimpleNamespace
import importlib
from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from app.main import create_app


def test_metrics_endpoint_fallback_returns_basic_metrics(monkeypatch):
    # Stub enterprise security middleware to no-op and relax settings
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

    # Force exporter to raise to exercise fallback path
    import app.core.prometheus_exporter as prometheus_exporter

    class _BrokenExporter:
        async def generate_metrics(self) -> str:  # pragma: no cover
            raise RuntimeError("boom")

    monkeypatch.setattr(prometheus_exporter, "get_prometheus_exporter", lambda: _BrokenExporter())

    app: FastAPI = create_app()
    client = TestClient(app, raise_server_exceptions=False)

    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert resp.headers.get("content-type") == "text/plain; version=0.0.4; charset=utf-8"
    body = resp.text
    assert "leanvibe_health_status" in body
    assert "leanvibe_uptime_seconds" in body
