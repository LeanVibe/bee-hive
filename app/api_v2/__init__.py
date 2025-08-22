"""
LeanVibe Agent Hive 2.0 - Consolidated API

Provides 15 resource-based endpoints with unified authentication,
error handling, and performance optimization for the autonomous
development platform.

Consolidates 96 API modules into 15 RESTful resource endpoints.
"""

from fastapi import APIRouter
from .middleware import auth_middleware, error_middleware, performance_middleware
from .routers import (
    agents,
    workflows, 
    tasks,
    projects,
    # coordination,  # Temporarily disabled due to missing schemas
    observability,
    security,
    resources,
    contexts,
    enterprise,
    websocket,
    health,
    admin,
    integrations,
    dashboard,
    plugins
)

# Create main API router
api_router = APIRouter(prefix="/api/v2")

# Note: APIRouter doesn't support middleware directly
# Middleware will be applied at the FastAPI app level in main.py

# Include all resource routers
api_router.include_router(agents.router, prefix="/agents", tags=["Agents"])
api_router.include_router(workflows.router, prefix="/workflows", tags=["Workflows"])
api_router.include_router(tasks.router, prefix="/tasks", tags=["Tasks"])
api_router.include_router(projects.router, prefix="/projects", tags=["Projects"])
# api_router.include_router(coordination.router, prefix="/coordination", tags=["Coordination"])  # Temporarily disabled
api_router.include_router(observability.router, prefix="/observability", tags=["Observability"])
api_router.include_router(security.router, prefix="/security", tags=["Security"])
api_router.include_router(resources.router, prefix="/resources", tags=["Resources"])
api_router.include_router(contexts.router, prefix="/contexts", tags=["Contexts"])
api_router.include_router(enterprise.router, prefix="/enterprise", tags=["Enterprise"])
api_router.include_router(websocket.router, prefix="/ws", tags=["WebSocket"])
api_router.include_router(health.router, prefix="/health", tags=["Health"])
api_router.include_router(admin.router, prefix="/admin", tags=["Admin"])
api_router.include_router(integrations.router, prefix="/integrations", tags=["Integrations"])
api_router.include_router(dashboard.router, prefix="/dashboard", tags=["Dashboard"])
api_router.include_router(plugins.router, tags=["Plugins"])

@api_router.get("/")
async def api_root():
    """API v2 root endpoint with consolidated resources."""
    return {
        "message": "LeanVibe Agent Hive 2.0 - Consolidated API",
        "version": "2.0.0",
        "modules_consolidated": "96 → 16 (83% reduction)",
        "resources": [
            "agents", "workflows", "tasks", "projects",  # "coordination" temporarily disabled
            "observability", "security", "resources", "contexts", "enterprise",
            "websocket", "health", "admin", "integrations", "dashboard", "plugins"
        ],
        "epic2_phase22": {
            "plugin_marketplace": "✅ Completed",
            "ai_discovery": "✅ Completed", 
            "security_certification": "✅ Completed",
            "developer_onboarding": "✅ Completed",
            "api_endpoints": "✅ Completed"
        },
        "docs": "/docs",
        "redoc": "/redoc"
    }

__all__ = ["api_router"]