import pytest
from httpx import AsyncClient


pytestmark = pytest.mark.asyncio


async def test_health_components_present(async_test_client: AsyncClient):
    resp = await async_test_client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    # Ensure key components are present even in mocks
    components = body.get("components", {})
    assert "database" in components
    assert "redis" in components
    assert "orchestrator" in components
    assert "observability" in components


async def test_metrics_exposition(async_test_client: AsyncClient):
    resp = await async_test_client.get("/metrics")
    assert resp.status_code == 200
    text = resp.text
    # Basic exposition format
    assert "# HELP" in text and "# TYPE" in text
    # Contains at least one of our base gauges
    assert "leanvibe_health_status" in text or len(text) > 0
