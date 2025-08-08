import pytest
from httpx import AsyncClient

pytestmark = pytest.mark.asyncio


async def test_dashboard_compat_transform_success(async_test_client: AsyncClient, monkeypatch):
    async def fake_get_dashboard_data():
        return {
            "metrics": {
                "active_projects": 2,
                "active_agents": 3,
                "agent_utilization": 66.6,
                "completed_tasks": 5,
                "active_conflicts": 1,
                "system_efficiency": 88.8,
                "system_status": "healthy",
                "last_updated": "2025-08-08T00:00:00Z",
            },
            "agent_activities": [
                {
                    "agent_id": "a1",
                    "name": "Agent One",
                    "status": "active",
                    "current_project": "P1",
                    "current_task": "T1",
                    "task_progress": 50,
                    "performance_score": 90,
                    "specializations": ["frontend"],
                }
            ],
            "project_snapshots": [
                {
                    "name": "P1",
                    "status": "active",
                    "progress_percentage": 75,
                    "participating_agents": ["a1"],
                    "completed_tasks": 3,
                    "active_tasks": 2,
                    "conflicts": 0,
                    "quality_score": 95,
                }
            ],
            "conflict_snapshots": [
                {
                    "conflict_type": "Resource",
                    "severity": "low",
                    "project_name": "P1",
                    "description": "desc",
                    "affected_agents": ["a1"],
                    "impact_score": 1,
                    "auto_resolvable": True,
                }
            ],
        }

    monkeypatch.setattr(
        "app.api.dashboard_compat.coordination_dashboard.get_dashboard_data",
        fake_get_dashboard_data,
    )

    resp = await async_test_client.get("/dashboard/api/live-data")
    assert resp.status_code == 200
    data = resp.json()
    assert data["metrics"]["active_projects"] == 2
    assert data["agent_activities"][0]["name"] == "Agent One"
    assert data["project_snapshots"][0]["name"] == "P1"
    assert data["conflict_snapshots"][0]["conflict_type"] == "Resource"


async def test_dashboard_compat_transform_fallback(async_test_client: AsyncClient, monkeypatch):
    async def boom():
        raise RuntimeError("fail")

    monkeypatch.setattr(
        "app.api.dashboard_compat.coordination_dashboard.get_dashboard_data",
        boom,
    )

    resp = await async_test_client.get("/dashboard/api/live-data")
    assert resp.status_code == 200
    data = resp.json()
    assert data["metrics"]["system_status"] == "degraded"
    assert isinstance(data["agent_activities"], list) and isinstance(data["project_snapshots"], list)
