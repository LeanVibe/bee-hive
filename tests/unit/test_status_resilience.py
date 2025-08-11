from types import SimpleNamespace
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.main import create_app


def test_status_endpoint_handles_backend_errors_gracefully(monkeypatch):
    # No-op security middleware; permissive settings
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

    # Break DB session generator
    def broken_get_async_session():
        async def agen():  # pragma: no cover
            raise RuntimeError("db boom")
            yield  # unreachable
        return agen()

    monkeypatch.setattr(db, "get_async_session", broken_get_async_session, raising=True)

    # Break Redis .info()
    class _BrokenRedis:
        async def info(self, *_args, **_kwargs):  # pragma: no cover
            raise RuntimeError("redis boom")

    monkeypatch.setattr(redis_mod, "get_redis", lambda: _BrokenRedis(), raising=True)

    app: FastAPI = create_app()
    client = TestClient(app, raise_server_exceptions=False)

    resp = client.get("/status")
    assert resp.status_code == 200
    body = resp.json()
    assert "timestamp" in body
    assert "version" in body
    assert isinstance(body.get("components"), dict)
    comps = body["components"]
    # Keys exist and are set to disconnected when backends fail
    assert "database" in comps and comps["database"].get("connected") is False
    assert "redis" in comps and comps["redis"].get("connected") is False
