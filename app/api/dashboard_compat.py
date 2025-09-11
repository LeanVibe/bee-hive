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
        raw_data = await coordination_dashboard.get_graph_data()

        # Extract stats from the graph data structure
        stats = raw_data.get("stats", {})
        nodes = raw_data.get("nodes", [])
        
        # Transform graph data into dashboard format
        agent_activities = []
        for node in nodes:
            if node.get("type") == "agent":
                metadata = node.get("metadata", {})
                agent_activities.append({
                    "agent_id": metadata.get("agent_id", node.get("id", "")),
                    "name": node.get("label", "Unknown Agent"),
                    "status": node.get("status", "unknown"),
                    "current_project": metadata.get("current_project"),
                    "current_task": metadata.get("current_task"),
                    "task_progress": 0.0,  # Not available in graph data
                    "performance_score": 0.0,  # Not available in graph data
                    "specializations": metadata.get("specializations", []),
                })

        live_data = {
            "metrics": {
                "active_projects": 0,  # Not tracked in current graph data
                "active_agents": stats.get("active_agents", 0),
                "agent_utilization": 0.0,  # Would need calculation
                "completed_tasks": 0,  # Not tracked in current graph data
                "active_conflicts": 0,  # Not tracked in current graph data
                "system_efficiency": 0.0,  # Would need calculation
                "system_status": "healthy" if stats.get("active_agents", 0) > 0 else "idle",
                "last_updated": raw_data.get("timestamp", datetime.utcnow().isoformat()),
            },
            "agent_activities": agent_activities,
            "project_snapshots": [],  # Not available in current graph data
            "conflict_snapshots": [],  # Not available in current graph data
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


