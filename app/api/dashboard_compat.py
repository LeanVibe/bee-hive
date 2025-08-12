"""
Dashboard Compatibility API (No HTML)

Provides legacy-compatible endpoints under `/dashboard/api/*` required by the
mobile PWA while keeping server-rendered views removed. These endpoints
transform internal dashboard data into the structure expected by the PWA.
"""

from datetime import datetime
from fastapi import APIRouter
import structlog

# Reuse the coordination dashboard's data aggregator without exposing its HTML routes
# Import only the data provider; legacy HTML routes remain unmounted
from ..core.coordination_dashboard import coordination_dashboard

logger = structlog.get_logger()

router = APIRouter(prefix="/dashboard/api", tags=["dashboard-compat"])  # legacy path expected by PWA


@router.get("/live-data")
async def get_live_data_compat():
    """Return live dashboard data in the format expected by the PWA.

    Path preserved for compatibility: `/dashboard/api/live-data`.
    """
    try:
        raw_data = await coordination_dashboard.get_dashboard_data()

        live_data = {
            "metrics": {
                "active_projects": raw_data.get("metrics", {}).get("active_projects", 0),
                "active_agents": raw_data.get("metrics", {}).get("active_agents", 0),
                "agent_utilization": raw_data.get("metrics", {}).get("agent_utilization", 0.0),
                "completed_tasks": raw_data.get("metrics", {}).get("completed_tasks", 0),
                "active_conflicts": raw_data.get("metrics", {}).get("active_conflicts", 0),
                "system_efficiency": raw_data.get("metrics", {}).get("system_efficiency", 0.0),
                "system_status": raw_data.get("metrics", {}).get("system_status", "healthy"),
                "last_updated": raw_data.get("metrics", {}).get("last_updated", datetime.utcnow().isoformat()),
            },
            "agent_activities": [
                {
                    "agent_id": agent.get("agent_id", ""),
                    "name": agent.get("name", "Unknown Agent"),
                    "status": agent.get("status", "unknown"),
                    "current_project": agent.get("current_project"),
                    "current_task": agent.get("current_task"),
                    "task_progress": agent.get("task_progress", 0.0),
                    "performance_score": agent.get("performance_score", 0.0),
                    "specializations": agent.get("specializations", []),
                }
                for agent in raw_data.get("agent_activities", [])
            ],
            "project_snapshots": [
                {
                    "name": project.get("name", "Unknown Project"),
                    "status": project.get("status", "unknown"),
                    "progress_percentage": project.get("progress_percentage", 0.0),
                    "participating_agents": project.get("participating_agents", []),
                    "completed_tasks": project.get("completed_tasks", 0),
                    "active_tasks": project.get("active_tasks", 0),
                    "conflicts": project.get("conflicts", 0),
                    "quality_score": project.get("quality_score", 0.0),
                }
                for project in raw_data.get("project_snapshots", [])
            ],
            "conflict_snapshots": [
                {
                    "conflict_type": conflict.get("conflict_type", "unknown"),
                    "severity": conflict.get("severity", "low"),
                    "project_name": conflict.get("project_name", "Unknown Project"),
                    "description": conflict.get("description", ""),
                    "affected_agents": conflict.get("affected_agents", []),
                    "impact_score": conflict.get("impact_score", 0.0),
                    "auto_resolvable": conflict.get("auto_resolvable", False),
                }
                for conflict in raw_data.get("conflict_snapshots", [])
            ],
        }

        return live_data

    except Exception as e:
        logger.error("Failed to get compat live data", error=str(e))
        return {
            "metrics": {
                "active_projects": 0,
                "active_agents": 0,
                "agent_utilization": 0.0,
                "completed_tasks": 0,
                "active_conflicts": 0,
                "system_efficiency": 0.0,
                "system_status": "degraded",
                "last_updated": datetime.utcnow().isoformat(),
            },
            "agent_activities": [],
            "project_snapshots": [],
            "conflict_snapshots": [],
        }


