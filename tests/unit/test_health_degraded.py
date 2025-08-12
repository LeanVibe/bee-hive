from types import SimpleNamespace
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.main import create_app


def test_health_reports_degraded_when_a_component_fails(monkeypatch):
    # No-op security and permissive settings
    import app.core.enterprise_security_system as ess
    import app.main as app_main
    import app.core.database as db
    import app.core.redis as redis_mod

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

    # Make DB check fail
    def broken_get_async_session():
        async def agen():  # pragma: no cover
            raise RuntimeError("db down")
            yield
        return agen()

    monkeypatch.setattr(db, "get_async_session", broken_get_async_session, raising=True)

    # Make Redis healthy
    class _FakeRedis:
        async def ping(self):  # pragma: no cover
            return True

    monkeypatch.setattr(redis_mod, "get_redis", lambda: _FakeRedis(), raising=True)

    app: FastAPI = create_app()
    # Ensure event_processor is absent to count as an unhealthy component
    if hasattr(app.state, "event_processor"):
        delattr(app.state, "event_processor")

    client = TestClient(app, raise_server_exceptions=False)
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body.get("status") == "degraded"
    comps = body.get("components", {})
    assert comps.get("database", {}).get("status") == "unhealthy"
    assert comps.get("redis", {}).get("status") == "healthy"
