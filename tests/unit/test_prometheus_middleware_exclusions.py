from types import SimpleNamespace
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.main import create_app


def test_prometheus_middleware_excludes_health_and_metrics(monkeypatch):
    # No-op security; permissive settings
    import app.core.enterprise_security_system as ess
    import app.main as app_main
    import app.core.prometheus_exporter as exporter_mod
    import app.observability.prometheus_middleware as pm

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

    calls = []

    class _SpyExporter:
        def record_http_request(self, **kwargs):  # pragma: no cover
            calls.append(kwargs)

        async def generate_metrics(self) -> str:  # pragma: no cover
            return "# HELP test 1\n# TYPE test counter\ntest 1\n"

    # Patch both the source module and the already-imported symbol in middleware
    monkeypatch.setattr(exporter_mod, "get_prometheus_exporter", lambda: _SpyExporter())
    monkeypatch.setattr(pm, "get_prometheus_exporter", lambda: _SpyExporter())

    app: FastAPI = create_app()
    client = TestClient(app, raise_server_exceptions=False)

    # Hitting excluded paths should NOT record
    client.get("/health")
    client.get("/metrics")

    # Hitting a normal path should record
    client.get("/status")

    assert calls, "expected non-empty calls after /status"
    # Ensure excluded paths didn't sneak in
    assert all(call["endpoint"] != "/health" for call in calls)
    assert all(call["endpoint"] != "/metrics" for call in calls)
