"""
API routes for LeanVibe Agent Hive 2.0

Aggregates all API endpoints for the multi-agent orchestration system.
Provides comprehensive REST API for agent management, task delegation,
and system monitoring.
"""

from fastapi import APIRouter

from .v1.agents import router as agents_router
from .v1.sessions import router as sessions_router  
from .v1.tasks import router as tasks_router
from .v1.workflows import router as workflows_router
from .v1.contexts import router as contexts_router
from .v1.system import router as system_router
from .v1.websocket import router as websocket_router
from .v1.workspaces import router as workspaces_router
from .v1.code_execution import router as code_router
from .v1.external_tools import router as external_tools_router
from .v1.coordination import router as coordination_router
from .v1.observability import router as observability_router
from .v1.sleep_wake import router as sleep_wake_router
from .v1.communication import router as communication_router
from .v1.github_integration import router as github_integration_router

# Main API router
router = APIRouter()

# Include all API version 1 routes
router.include_router(agents_router, prefix="/agents", tags=["agents"])
router.include_router(sessions_router, prefix="/sessions", tags=["sessions"])
router.include_router(tasks_router, prefix="/tasks", tags=["tasks"])
router.include_router(workflows_router, prefix="/workflows", tags=["workflows"])
router.include_router(contexts_router, prefix="/contexts", tags=["contexts"])
router.include_router(system_router, prefix="/system", tags=["system"])
router.include_router(websocket_router, prefix="/ws", tags=["websocket"])
router.include_router(workspaces_router, prefix="/workspaces", tags=["workspaces"])
router.include_router(code_router, prefix="/code", tags=["code-execution"])
router.include_router(external_tools_router, prefix="/tools", tags=["external-tools"])
router.include_router(coordination_router, prefix="/coordination", tags=["multi-agent-coordination"])
router.include_router(observability_router, tags=["observability"])
router.include_router(sleep_wake_router, tags=["sleep-wake"])
router.include_router(communication_router, tags=["communication"])
router.include_router(github_integration_router, tags=["github-integration"])


@router.get("/")
async def api_root():
    """API root endpoint."""
    return {
        "message": "LeanVibe Agent Hive 2.0 API",
        "version": "2.0.0",
        "docs": "/docs",
        "redoc": "/redoc"
    }