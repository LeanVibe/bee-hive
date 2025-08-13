import pytest
from httpx import AsyncClient

pytestmark = pytest.mark.asyncio


async def test_dashboard_live_data_compat_structure(async_test_client: AsyncClient):
    resp = await async_test_client.get("/dashboard/api/live-data")
    assert resp.status_code == 200
    data = resp.json()
    # Ensure core sections exist (helps cover transform path)
    assert "metrics" in data
    assert "agent_activities" in data
    assert "project_snapshots" in data
    assert "conflict_snapshots" in data
