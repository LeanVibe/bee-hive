"""
Epic B API v2 Module

Core agent management API endpoints for the Mobile PWA.
Implements the fundamental agent lifecycle operations required
for demonstrable multi-agent coordination with real-time updates.
"""

from fastapi import APIRouter
from .agents import router as agents_router
from .tasks import router as tasks_router
from .websockets import router as websockets_router

# Main v2 API router
api_router = APIRouter()

# Include agent management endpoints
api_router.include_router(agents_router)

# Include task management endpoints  
api_router.include_router(tasks_router)

# Include WebSocket endpoints for real-time updates
api_router.include_router(websockets_router)