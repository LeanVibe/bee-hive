import pytest
from httpx import AsyncClient

pytestmark = pytest.mark.asyncio


async def test_status_shape(async_test_client: AsyncClient):
    resp = await async_test_client.get("/status")
    assert resp.status_code == 200
    body = resp.json()
    assert "timestamp" in body
    assert "version" in body
    assert "components" in body and isinstance(body["components"], dict)
    comps = body["components"]
    # Ensure keys exist regardless of connection state
    assert "database" in comps
    assert "redis" in comps
    assert "orchestrator" in comps
    assert "observability" in comps
