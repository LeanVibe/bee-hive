"""
AgentManagementAPI v2 - Unified Agent Lifecycle and Coordination Management

Consolidated agent management module implementing Phase 3 of Epic 4 API Architecture Consolidation.
Unifies 6 agent management modules into a single, high-performance agent management API.

Consolidated Modules:
- app/api/v1/coordination.py -> Agent coordination and multi-agent project management
- app/api/endpoints/agents.py -> Core agent lifecycle operations
- app/api/v2/agents.py -> Epic B agent management API
- app/api/agent_activation.py -> Agent activation and system control
- app/api/v1/agents_simple.py -> Simple agent CRUD operations
- app/core/subagent_coordination.py -> Agent coordination utilities

Key Features:
- <200ms response times with intelligent caching and optimization
- Multi-agent coordination and task routing with conflict resolution
- Agent lifecycle management (create, activate, deactivate, monitor)
- Real-time WebSocket coordination updates with <50ms latency
- Integration with Epic 1 ConsolidatedProductionOrchestrator
- OAuth2 + RBAC security with comprehensive audit logging
- Full backwards compatibility with v1 agent management endpoints
- Agent activation control for system operational status
- Comprehensive agent performance analytics and health monitoring

Architecture:
- core.py: Main agent management endpoints and lifecycle operations
- coordination.py: Multi-agent coordination, task routing, and conflict resolution
- lifecycle.py: Agent activation, deactivation, and system control
- models.py: Unified data models for agents, coordination, and status
- middleware.py: Security, validation, caching, and performance middleware
- utils.py: Shared agent utilities, helpers, and common operations
- compatibility.py: v1 API backwards compatibility layer

Performance Targets:
- Agent operation response time: <200ms
- Coordination updates latency: <50ms
- Multi-agent task routing: <100ms
- Agent activation time: <5s
- System status queries: <50ms
"""

from .core import router as agents_router
from .models import *
from .middleware import *
from .utils import *

__all__ = [
    "agents_router",
    "AgentResponse",
    "AgentListResponse", 
    "AgentStatsResponse",
    "AgentCoordinationResponse",
    "AgentActivationResponse",
    "ProjectResponse",
    "ConflictResponse"
]