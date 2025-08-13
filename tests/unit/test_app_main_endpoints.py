import pytest
from httpx import AsyncClient

pytestmark = pytest.mark.asyncio


async def test_status_endpoint_returns_json(async_test_client: AsyncClient):
    resp = await async_test_client.get("/status")
    assert resp.status_code == 200
    data = resp.json()
    assert "version" in data
    assert "components" in data


async def test_metrics_endpoint_returns_prometheus_text(async_test_client: AsyncClient):
    resp = await async_test_client.get("/metrics")
    assert resp.status_code == 200
    assert resp.headers.get("content-type", "").startswith(
        "text/plain"
    )
    body = resp.text
    assert "# HELP" in body or "leanvibe_" in body


async def test_global_exception_handler_returns_500(async_test_client: AsyncClient, test_app):
    from fastapi import APIRouter
    from httpx import AsyncClient as _AsyncClient
    from httpx import ASGITransport as _ASGITransport

    router = APIRouter()

    @router.get("/_boom")
    async def boom():  # type: ignore[unused-ignore]
        raise ValueError("boom")

    # Mount a temporary crashing route to trigger global handler
    test_app.include_router(router)

    # Use transport with raise_app_exceptions=False to assert 500 response body
    transport = _ASGITransport(app=test_app, raise_app_exceptions=False)
    async with _AsyncClient(transport=transport, base_url="http://localhost:8000") as client:
        resp = await client.get("/_boom")
        assert resp.status_code == 500
        data = resp.json()
        assert data.get("error") == "Internal server error"
