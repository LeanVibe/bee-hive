import pytest
from httpx import AsyncClient

pytestmark = pytest.mark.asyncio


async def test_dashboard_compat_transform_success(async_test_client: AsyncClient, monkeypatch):
    async def fake_get_graph_data():
        return {
            "nodes": [
                {
                    "id": "agent_a1",
                    "label": "Agent One",
                    "type": "agent",
                    "status": "active",
                    "metadata": {
                        "agent_id": "a1",
                        "session_id": "session1",
                        "agent_type": "backend-engineer",
                        "current_project": "P1",
                        "current_task": "T1",
                        "specializations": ["frontend"]
                    }
                }
            ],
            "edges": [],
            "stats": {
                "total_nodes": 1,
                "total_edges": 0,
                "active_agents": 3,
                "tools_used": 0,
                "contexts_shared": 0
            },
            "session_colors": {},
            "timestamp": "2025-08-08T00:00:00Z"
        }

    monkeypatch.setattr(
        "app.api.dashboard_compat.coordination_dashboard.get_graph_data",
        fake_get_graph_data,
    )

    resp = await async_test_client.get("/dashboard/api/live-data")
    assert resp.status_code == 200
    data = resp.json()
    assert data["metrics"]["active_agents"] == 3
    assert data["metrics"]["system_status"] == "healthy"
    assert data["agent_activities"][0]["name"] == "Agent One"
    assert data["agent_activities"][0]["agent_id"] == "a1"
    assert data["project_snapshots"] == []
    assert data["conflict_snapshots"] == []


async def test_dashboard_compat_transform_fallback(async_test_client: AsyncClient, monkeypatch):
    async def boom():
        raise RuntimeError("fail")

    monkeypatch.setattr(
        "app.api.dashboard_compat.coordination_dashboard.get_graph_data",
        boom,
    )

    resp = await async_test_client.get("/dashboard/api/live-data")
    assert resp.status_code == 200
    data = resp.json()
    assert data["metrics"]["system_status"] == "degraded"
    assert isinstance(data["agent_activities"], list) and isinstance(data["project_snapshots"], list)
