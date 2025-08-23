"""
API routes for LeanVibe Agent Hive 2.0

Aggregates all API endpoints for the multi-agent orchestration system.
Provides comprehensive REST API for agent management, task delegation,
and system monitoring.
"""

from fastapi import APIRouter
from fastapi import Depends

from ..core.auth import Permission
from ..core.auth import get_current_user
from ..core.auth import require_permission as require_basic_permission
from .auth_endpoints import router as auth_router

# Import working API endpoints
from .enterprise_pilots import router as pilots_router
from .v1.coordination_monitoring import router as coordination_monitoring_router
from .v1.github_integration import router as github_router
from .v1.websocket import router as websocket_router
from .project_index import router as project_index_router

# EPIC 1 PHASE 1.1: Add missing agent API endpoints (simplified version for CLI integration)
from .v1.agents_simple import router as agents_router

# Temporarily disabled to avoid model conflicts
# from .coordination_endpoints import router as coordination_router

# Main API router
router = APIRouter()

# Include working API routes
router.include_router(auth_router)
router.include_router(pilots_router)
router.include_router(websocket_router, prefix="/ws")
router.include_router(github_router, prefix="/api/v1")
router.include_router(coordination_monitoring_router, prefix="/api/v1")
router.include_router(project_index_router)  # Project Index API

# EPIC 1 PHASE 1.1: Include agent API endpoints for CLI integration
router.include_router(agents_router, prefix="/agents", tags=["agents"])

# router.include_router(coordination_router)  # Temporarily disabled


@router.get("/")
async def api_root():
    """API root endpoint."""
    return {
        "message": "LeanVibe Agent Hive 2.0 API",
        "version": "2.0.0",
        "docs": "/docs",
        "redoc": "/redoc"
    }


@router.get("/protected/ping")
async def protected_ping(current_user=Depends(get_current_user)):
    return {"pong": True, "user_id": getattr(current_user, 'id', None)}


@router.get("/protected/admin")
async def protected_admin_route(
    _=Depends(require_basic_permission(Permission.MANAGE_USERS))
):
    return {"ok": True, "route": "admin"}
