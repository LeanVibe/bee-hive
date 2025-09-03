"""
TaskExecutionAPI - Epic 4 Phase 4 Consolidated Task Execution Module

Unified task execution API providing comprehensive workflow orchestration,
intelligent scheduling, and team coordination capabilities.

Consolidates 6 source files (4,394 lines) into unified architecture following
successful Phase 2-3 patterns achieving 94.4% consolidation efficiency.

Architecture:
- core.py: Main task execution endpoints
- workflows.py: Workflow orchestration logic  
- scheduling.py: Intelligent scheduling algorithms
- orchestration.py: Task delegation and execution management
- models.py: Unified data models and schemas
- middleware.py: Authentication, validation, rate limiting
- utils.py: Shared utilities and helpers
- compatibility.py: v1 API backwards compatibility

Performance Targets:
- <200ms task creation response time
- <500ms workflow initiation response time  
- <50ms status query response time
- 94.4% consolidation efficiency (target from Phase 2-3 success)

Security Features:
- OAuth2 + task-specific permissions
- Comprehensive audit logging
- Rate limiting and DDoS protection
- Input validation and sanitization

Integration:
- Deep Epic 1 ConsolidatedProductionOrchestrator integration
- Workflow engine state management
- Redis coordination and caching
- WebSocket real-time updates
"""

from fastapi import APIRouter
from .core import router as core_router
from .workflows import router as workflows_router
from .scheduling import router as scheduling_router

# Create main tasks router with consolidated endpoints
router = APIRouter(prefix="/tasks", tags=["Task Execution API"])

# Include sub-routers with proper prefixes
router.include_router(core_router, prefix="")
router.include_router(workflows_router, prefix="/workflows")  
router.include_router(scheduling_router, prefix="/scheduling")

# Export key components for external use
from .models import (
    TaskExecutionRequest,
    TaskExecutionResponse, 
    WorkflowExecutionRequest,
    WorkflowExecutionResponse,
    ScheduleRequest,
    ScheduleResponse
)

from .core import (
    create_task,
    assign_task, 
    update_task_status,
    cancel_task,
    update_task_priority
)

from .workflows import (
    create_workflow,
    execute_workflow,
    get_workflow_progress,
    pause_workflow,
    resume_workflow
)

from .scheduling import (
    analyze_patterns,
    optimize_schedule,
    resolve_conflicts,
    get_predictive_forecast
)

__all__ = [
    "router",
    # Models
    "TaskExecutionRequest", 
    "TaskExecutionResponse",
    "WorkflowExecutionRequest",
    "WorkflowExecutionResponse", 
    "ScheduleRequest",
    "ScheduleResponse",
    # Core endpoints
    "create_task",
    "assign_task",
    "update_task_status", 
    "cancel_task",
    "update_task_priority",
    # Workflow endpoints
    "create_workflow",
    "execute_workflow",
    "get_workflow_progress",
    "pause_workflow", 
    "resume_workflow",
    # Scheduling endpoints
    "analyze_patterns",
    "optimize_schedule", 
    "resolve_conflicts",
    "get_predictive_forecast"
]