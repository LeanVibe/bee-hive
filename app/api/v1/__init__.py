"""
API v1 router for LeanVibe Agent Hive 2.0

Consolidates all v1 endpoints that the frontend expects.
Provides proper REST API structure for frontend-backend integration.
"""

from fastapi import APIRouter

# Import basic endpoint modules (avoiding complex dependencies for now)
from .system import router as system_router

# Create main v1 router
router = APIRouter()

# Include working sub-routers
router.include_router(system_router, prefix="/system", tags=["system"])

# Create simple agent endpoints for frontend compatibility
@router.get("/agents")
async def list_agents():
    """List all agents - simplified endpoint for frontend."""
    return {
        "agents": [],
        "total": 0,
        "offset": 0,
        "limit": 50
    }

@router.post("/agents")
async def create_agent(agent_data: dict):
    """Create a new agent - simplified endpoint for frontend."""
    return {
        "id": "agent-123",
        "name": agent_data.get("name", "new-agent"),
        "status": "created",
        "message": "Agent created successfully"
    }

@router.get("/agents/{agent_id}")
async def get_agent(agent_id: str):
    """Get agent by ID - simplified endpoint for frontend."""
    return {
        "id": agent_id,
        "name": f"Agent {agent_id}",
        "status": "active",
        "type": "claude",
        "created_at": "2024-01-01T00:00:00Z"
    }

# Create simple task endpoints for frontend compatibility
@router.get("/tasks")
async def list_tasks():
    """List all tasks - simplified endpoint for frontend."""
    return {
        "tasks": [],
        "total": 0,
        "offset": 0,
        "limit": 50
    }

@router.post("/tasks")
async def create_task(task_data: dict):
    """Create a new task - simplified endpoint for frontend."""
    return {
        "id": "task-123",
        "title": task_data.get("title", "New Task"),
        "status": "created",
        "message": "Task created successfully"
    }

@router.get("/tasks/{task_id}")
async def get_task(task_id: str):
    """Get task by ID - simplified endpoint for frontend."""
    return {
        "id": task_id,
        "title": f"Task {task_id}",
        "status": "pending",
        "priority": "medium",
        "created_at": "2024-01-01T00:00:00Z"
    }

@router.get("/")
async def v1_root():
    """API v1 root endpoint."""
    return {
        "message": "LeanVibe Agent Hive API v1",
        "version": "1.0.0",
        "endpoints": {
            "system": "/api/v1/system",
            "agents": "/api/v1/agents", 
            "tasks": "/api/v1/tasks"
        }
    }

__all__ = ["router"]
