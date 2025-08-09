import pytest
from httpx import AsyncClient

pytestmark = pytest.mark.asyncio


async def test_legacy_dashboard_routes_not_mounted(async_test_client: AsyncClient):
    # Legacy server-rendered dashboard paths should not exist
    for path in ("/dashboard", "/dashboard/"):
        resp = await async_test_client.get(path)
        assert resp.status_code in (404, 405)


async def test_legacy_ws_endpoints_not_mounted(async_test_client: AsyncClient):
    # Old v1 websocket endpoints should not be mounted in the app
    for path in (
        "/api/v1/observability_websocket",
        "/api/v1/websocket",
    ):
        resp = await async_test_client.get(path)
        assert resp.status_code in (404, 405)
