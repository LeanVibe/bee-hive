"""
Consolidated API routers for LeanVibe Agent Hive 2.0

Provides 16 resource-based endpoints that consolidate
96 original API modules into a clean, RESTful architecture.
Includes Epic 2 Phase 2.2 Plugin Marketplace & Discovery.
"""

# Import all routers for easy access
from . import (
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

__all__ = [
    "agents",
    "workflows", 
    "tasks",
    "projects",
    # "coordination",  # Temporarily disabled
    "observability",
    "security",
    "resources",
    "contexts",
    "enterprise",
    "websocket",
    "health",
    "admin",
    "integrations",
    "dashboard",
    "plugins"
]