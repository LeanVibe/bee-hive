"""
Consolidated API routers for LeanVibe Agent Hive 2.0

Provides 15 resource-based endpoints that consolidate
96 original API modules into a clean, RESTful architecture.
"""

# Import all routers for easy access
from . import (
    agents,
    workflows,
    tasks, 
    projects,
    coordination,
    observability,
    security,
    resources,
    contexts,
    enterprise,
    websocket,
    health,
    admin,
    integrations,
    dashboard
)

__all__ = [
    "agents",
    "workflows", 
    "tasks",
    "projects",
    "coordination",
    "observability",
    "security",
    "resources",
    "contexts",
    "enterprise",
    "websocket",
    "health",
    "admin",
    "integrations",
    "dashboard"
]