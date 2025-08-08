import pytest
from httpx import AsyncClient

pytestmark = pytest.mark.asyncio

async def test_health_endpoint(async_test_client: AsyncClient):
    resp = await async_test_client.get('/health')
    assert resp.status_code == 200
    data = resp.json()
    assert data['status'] in {'healthy', 'degraded', 'unhealthy'}

async def test_live_data_compat(async_test_client: AsyncClient):
    resp = await async_test_client.get('/dashboard/api/live-data')
    assert resp.status_code == 200
    data = resp.json()
    assert 'metrics' in data
    assert 'agent_activities' in data
