"""
API routes for LeanVibe Agent Hive 2.0

Aggregates all API endpoints for the multi-agent orchestration system.
Provides comprehensive REST API for agent management, task delegation,
and system monitoring.
"""

from fastapi import APIRouter

# Import working API endpoints
from .enterprise_pilots import router as pilots_router
from ..core.auth import auth_router

# Main API router
router = APIRouter()

# Include working API routes
router.include_router(auth_router)
router.include_router(pilots_router)


@router.get("/")
async def api_root():
    """API root endpoint."""
    return {
        "message": "LeanVibe Agent Hive 2.0 API",
        "version": "2.0.0",
        "docs": "/docs",
        "redoc": "/redoc"
    }