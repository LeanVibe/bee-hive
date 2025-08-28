"""
Task Management API Endpoints - Epic C Phase 1

Core API endpoints for task creation, assignment, progress tracking, and lifecycle management.
Follows established FastAPI patterns with full orchestrator integration for production usage.

Key Features:
- Task creation and assignment with capability matching
- Real-time progress tracking and status updates
- Priority management and task cancellation
- Integration with orchestrator for agent assignment
- Response times <200ms per Epic requirements
- Comprehensive error handling and logging

Epic C Phase 1: API Endpoint Implementation
"""

import asyncio
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any

from fastapi import APIRouter, HTTPException, Depends, Query, Path, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ...schemas.task import (
    TaskCreate, TaskUpdate, TaskResponse, TaskListResponse,
    TaskAssignmentRequest, TaskStatsResponse
)
from ...models.task import TaskStatus, TaskPriority, TaskType
from ...core.database import get_async_session
from ...core.logging_service import get_component_logger
from ...core.simple_orchestrator import SimpleOrchestrator

# Initialize logging
logger = get_component_logger("tasks_api")

# Create router
router = APIRouter(prefix="/api/v1/tasks", tags=["tasks"])


class TaskOperationResponse(BaseModel):
    """Standard response model for task operations."""
    success: bool = True
    message: str
    task_id: Optional[str] = None
    agent_id: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class TaskStatusResponse(BaseModel):
    """Response model for task status queries."""
    task_id: str
    status: TaskStatus
    progress: float = Field(ge=0.0, le=1.0, description="Task completion progress (0.0-1.0)")
    assigned_agent_id: Optional[str] = None
    started_at: Optional[datetime] = None
    estimated_completion: Optional[datetime] = None
    last_update: datetime


class TaskPriorityUpdateRequest(BaseModel):
    """Request model for task priority updates."""
    priority: TaskPriority = Field(..., description="New task priority")
    reason: Optional[str] = Field(None, description="Reason for priority change")


async def get_orchestrator() -> SimpleOrchestrator:
    """Dependency to get orchestrator instance."""
    try:
        # Try to get or create orchestrator instance
        # TODO: Replace with proper dependency injection
        orchestrator = SimpleOrchestrator()
        await orchestrator.initialize()
        return orchestrator
    except Exception as e:
        logger.warning(f"Orchestrator unavailable: {e}")
        raise HTTPException(
            status_code=503,
            detail="Task orchestration service temporarily unavailable"
        )


@router.post("/", response_model=TaskResponse, status_code=201)
async def create_task(
    task_data: TaskCreate,
    background_tasks: BackgroundTasks,
    assign_immediately: bool = Query(False, description="Attempt immediate agent assignment"),
    orchestrator: SimpleOrchestrator = Depends(get_orchestrator)
):
    """
    Create a new task and optionally assign it to an available agent.
    
    Creates a task with the specified requirements and capabilities.
    If assign_immediately is True, attempts to find and assign a suitable agent.
    
    **Performance Target**: <200ms response time
    **Integration**: Full orchestrator integration for agent assignment
    """
    try:
        start_time = datetime.utcnow()
        
        # Generate unique task ID
        task_id = str(uuid.uuid4())
        
        # Create task configuration
        task_config = {
            "id": task_id,
            "title": task_data.title,
            "description": task_data.description,
            "task_type": task_data.task_type,
            "priority": task_data.priority,
            "required_capabilities": task_data.required_capabilities or [],
            "estimated_effort": task_data.estimated_effort,
            "context": task_data.context or {},
            "status": TaskStatus.PENDING,
            "created_at": start_time.isoformat(),
            "updated_at": start_time.isoformat()
        }
        
        # Create task in orchestrator
        try:
            orchestrator_result = await orchestrator.create_task(task_config)
            logger.info(f"Task {task_data.title} created in orchestrator: {orchestrator_result}")
        except Exception as e:
            logger.warning(f"Orchestrator task creation failed for {task_data.title}: {e}")
            # Continue with creation but log the issue
        
        # Attempt immediate assignment if requested
        assigned_agent_id = None
        if assign_immediately:
            try:
                assignment_result = await orchestrator.assign_task(
                    task_id=task_id,
                    required_capabilities=task_data.required_capabilities or []
                )
                if assignment_result.get("success"):
                    assigned_agent_id = assignment_result.get("agent_id")
                    task_config["status"] = TaskStatus.ASSIGNED
                    task_config["assigned_agent_id"] = assigned_agent_id
                    task_config["assigned_at"] = datetime.utcnow().isoformat()
                    logger.info(f"Task {task_id} assigned to agent {assigned_agent_id}")
                else:
                    logger.info(f"No suitable agent found for immediate assignment of task {task_id}")
            except Exception as e:
                logger.warning(f"Immediate assignment failed for task {task_id}: {e}")
        
        # Prepare response data
        response_data = TaskResponse(
            id=uuid.UUID(task_id),
            title=task_data.title,
            description=task_data.description,
            task_type=task_data.task_type,
            status=task_config["status"],
            priority=task_data.priority,
            assigned_agent_id=assigned_agent_id,
            required_capabilities=task_data.required_capabilities or [],
            estimated_effort=task_data.estimated_effort,
            actual_effort=None,
            result=None,
            error_message=None,
            retry_count=0,
            created_at=start_time,
            assigned_at=datetime.fromisoformat(task_config["assigned_at"]) if task_config.get("assigned_at") else None,
            started_at=None,
            completed_at=None
        )
        
        # Performance monitoring
        elapsed_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        if elapsed_ms > 200:
            logger.warning(f"Task creation took {elapsed_ms:.1f}ms (target: <200ms)")
        
        logger.info(f"✅ Created task: {task_data.title} ({task_id}) in {elapsed_ms:.1f}ms")
        return response_data
        
    except Exception as e:
        logger.error(f"Failed to create task {task_data.title}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Task creation failed: {str(e)}"
        )


@router.get("/", response_model=TaskListResponse)
async def list_tasks(
    status: Optional[TaskStatus] = Query(None, description="Filter by task status"),
    priority: Optional[TaskPriority] = Query(None, description="Filter by task priority"),
    assigned_agent_id: Optional[str] = Query(None, description="Filter by assigned agent"),
    limit: int = Query(20, ge=1, le=100, description="Maximum number of tasks to return"),
    offset: int = Query(0, ge=0, description="Number of tasks to skip"),
    orchestrator: SimpleOrchestrator = Depends(get_orchestrator)
):
    """
    List all tasks with optional filtering.
    
    Retrieves tasks from the orchestrator with current status and assignment information.
    Supports pagination and filtering for efficient querying.
    """
    try:
        start_time = datetime.utcnow()
        
        # Get tasks from orchestrator
        try:
            tasks_data = await orchestrator.list_tasks(
                status_filter=status,
                priority_filter=priority,
                agent_filter=assigned_agent_id,
                limit=limit,
                offset=offset
            )
        except Exception as e:
            logger.warning(f"Orchestrator task query failed: {e}")
            # Return empty list if orchestrator unavailable
            tasks_data = {"tasks": [], "total": 0}
        
        # Convert to response format
        tasks = []
        for task_data in tasks_data.get("tasks", []):
            try:
                task_response = TaskResponse(
                    id=uuid.UUID(task_data["id"]),
                    title=task_data["title"],
                    description=task_data.get("description"),
                    task_type=task_data.get("task_type"),
                    status=task_data.get("status", TaskStatus.PENDING),
                    priority=task_data.get("priority", TaskPriority.MEDIUM),
                    assigned_agent_id=task_data.get("assigned_agent_id"),
                    required_capabilities=task_data.get("required_capabilities", []),
                    estimated_effort=task_data.get("estimated_effort"),
                    actual_effort=task_data.get("actual_effort"),
                    result=task_data.get("result"),
                    error_message=task_data.get("error_message"),
                    retry_count=task_data.get("retry_count", 0),
                    created_at=task_data.get("created_at"),
                    assigned_at=task_data.get("assigned_at"),
                    started_at=task_data.get("started_at"),
                    completed_at=task_data.get("completed_at")
                )
                tasks.append(task_response)
            except Exception as e:
                logger.warning(f"Skipping malformed task data: {e}")
                continue
        
        response = TaskListResponse(
            tasks=tasks,
            total=tasks_data.get("total", len(tasks)),
            offset=offset,
            limit=limit
        )
        
        # Performance monitoring
        elapsed_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        logger.info(f"Listed {len(tasks)} tasks in {elapsed_ms:.1f}ms")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to list tasks: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Task listing failed: {str(e)}"
        )


@router.get("/{task_id}/status", response_model=TaskStatusResponse)
async def get_task_status(
    task_id: str = Path(..., description="Task ID"),
    orchestrator: SimpleOrchestrator = Depends(get_orchestrator)
):
    """
    Get real-time status and progress information for a specific task.
    
    Provides current task status, progress percentage, and estimated completion time
    from the orchestrator's task tracking system.
    """
    try:
        # Validate UUID format
        try:
            uuid.UUID(task_id)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="Invalid task ID format"
            )
        
        # Get task status from orchestrator
        try:
            status_data = await orchestrator.get_task_status(task_id)
        except Exception as e:
            logger.warning(f"Orchestrator status query failed for task {task_id}: {e}")
            raise HTTPException(
                status_code=404,
                detail="Task status not available"
            )
        
        if not status_data:
            raise HTTPException(
                status_code=404,
                detail="Task not found"
            )
        
        response = TaskStatusResponse(
            task_id=task_id,
            status=status_data.get("status", TaskStatus.UNKNOWN),
            progress=status_data.get("progress", 0.0),
            assigned_agent_id=status_data.get("assigned_agent_id"),
            started_at=status_data.get("started_at"),
            estimated_completion=status_data.get("estimated_completion"),
            last_update=status_data.get("last_update", datetime.utcnow())
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get status for task {task_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Task status retrieval failed: {str(e)}"
        )


@router.put("/{task_id}/priority", response_model=TaskOperationResponse)
async def update_task_priority(
    task_id: str = Path(..., description="Task ID"),
    priority_update: TaskPriorityUpdateRequest = ...,
    orchestrator: SimpleOrchestrator = Depends(get_orchestrator)
):
    """
    Update the priority of a specific task.
    
    Changes task priority in the orchestrator's queue, potentially affecting
    assignment order and execution scheduling.
    """
    try:
        # Validate UUID format
        try:
            uuid.UUID(task_id)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="Invalid task ID format"
            )
        
        # Update priority through orchestrator
        try:
            result = await orchestrator.update_task_priority(
                task_id=task_id,
                new_priority=priority_update.priority,
                reason=priority_update.reason
            )
        except Exception as e:
            logger.warning(f"Orchestrator priority update failed for task {task_id}: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Priority update failed: {str(e)}"
            )
        
        if not result.get("success", False):
            raise HTTPException(
                status_code=400,
                detail=result.get("message", "Priority update failed")
            )
        
        response = TaskOperationResponse(
            success=True,
            message=f"Task priority updated to {priority_update.priority.value}",
            task_id=task_id
        )
        
        logger.info(f"✅ Updated task {task_id} priority to {priority_update.priority.value}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update task {task_id} priority: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Priority update failed: {str(e)}"
        )


@router.delete("/{task_id}", response_model=TaskOperationResponse)
async def cancel_task(
    task_id: str = Path(..., description="Task ID"),
    force: bool = Query(False, description="Force cancellation even if task is running"),
    reason: Optional[str] = Query(None, description="Reason for cancellation"),
    orchestrator: SimpleOrchestrator = Depends(get_orchestrator)
):
    """
    Cancel a task and remove it from the system.
    
    Cancels task execution and removes it from the orchestrator's queue.
    Optionally supports force cancellation for emergency situations.
    """
    try:
        # Validate UUID format
        try:
            uuid.UUID(task_id)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="Invalid task ID format"
            )
        
        # Cancel task through orchestrator
        try:
            result = await orchestrator.cancel_task(
                task_id=task_id,
                force=force,
                reason=reason
            )
        except Exception as e:
            logger.warning(f"Orchestrator cancellation failed for task {task_id}: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Task cancellation failed: {str(e)}"
            )
        
        if not result.get("success", False):
            raise HTTPException(
                status_code=400,
                detail=result.get("message", "Task cancellation failed")
            )
        
        response = TaskOperationResponse(
            success=True,
            message=f"Task {task_id} cancelled successfully",
            task_id=task_id
        )
        
        logger.info(f"✅ Cancelled task {task_id}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cancel task {task_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Task cancellation failed: {str(e)}"
        )


@router.post("/{task_id}/assign", response_model=TaskOperationResponse)
async def assign_task(
    task_id: str = Path(..., description="Task ID"),
    assignment_request: TaskAssignmentRequest = ...,
    orchestrator: SimpleOrchestrator = Depends(get_orchestrator)
):
    """
    Assign a task to a specific agent or find the best available agent.
    
    If agent_id is specified, assigns the task to that specific agent.
    Otherwise, finds the best available agent based on capabilities and workload.
    """
    try:
        # Validate UUID format
        try:
            uuid.UUID(task_id)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="Invalid task ID format"
            )
        
        # Validate agent ID if provided
        if assignment_request.agent_id:
            try:
                uuid.UUID(assignment_request.agent_id)
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid agent ID format"
                )
        
        # Assign task through orchestrator
        try:
            result = await orchestrator.assign_task(
                task_id=task_id,
                agent_id=assignment_request.agent_id,
                priority_override=assignment_request.priority_override,
                context_override=assignment_request.context_override
            )
        except Exception as e:
            logger.warning(f"Orchestrator assignment failed for task {task_id}: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Task assignment failed: {str(e)}"
            )
        
        if not result.get("success", False):
            raise HTTPException(
                status_code=400,
                detail=result.get("message", "Task assignment failed")
            )
        
        assigned_agent_id = result.get("agent_id")
        response = TaskOperationResponse(
            success=True,
            message=f"Task assigned to agent {assigned_agent_id}",
            task_id=task_id,
            agent_id=assigned_agent_id
        )
        
        logger.info(f"✅ Assigned task {task_id} to agent {assigned_agent_id}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to assign task {task_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Task assignment failed: {str(e)}"
        )


@router.get("/stats", response_model=TaskStatsResponse)
async def get_task_stats(
    orchestrator: SimpleOrchestrator = Depends(get_orchestrator)
):
    """
    Get comprehensive statistics about the task system.
    
    Provides overall task metrics including completion rates, performance,
    and current system utilization.
    """
    try:
        # Get stats from orchestrator
        try:
            stats = await orchestrator.get_task_stats()
        except Exception as e:
            logger.warning(f"Orchestrator stats query failed: {e}")
            raise HTTPException(
                status_code=503,
                detail="Task statistics temporarily unavailable"
            )
        
        response = TaskStatsResponse(
            total_tasks=stats.get("total_tasks", 0),
            pending_tasks=stats.get("pending_tasks", 0),
            running_tasks=stats.get("running_tasks", 0),
            completed_tasks=stats.get("completed_tasks", 0),
            failed_tasks=stats.get("failed_tasks", 0),
            average_completion_time_minutes=stats.get("average_completion_time_minutes", 0.0),
            success_rate=stats.get("success_rate", 0.0),
            active_agents=stats.get("active_agents", 0)
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get task statistics: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Statistics retrieval failed: {str(e)}"
        )


# Health check endpoint for tasks subsystem
@router.get("/health/status")
async def tasks_health_check():
    """Health check endpoint for the tasks subsystem."""
    try:
        # Test orchestrator connectivity
        orchestrator_healthy = True
        orchestrator_error = None
        task_queue_size = 0
        
        try:
            orchestrator = SimpleOrchestrator()
            await orchestrator.initialize()
            # Basic health check operations
            health_info = await orchestrator.get_system_health()
            task_queue_size = health_info.get("task_queue_size", 0)
        except Exception as e:
            orchestrator_healthy = False
            orchestrator_error = str(e)
        
        health_status = {
            "service": "tasks_api",
            "healthy": orchestrator_healthy,
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": {
                "task_queue_size": task_queue_size
            },
            "components": {
                "orchestrator": {
                    "healthy": orchestrator_healthy,
                    "error": orchestrator_error
                }
            }
        }
        
        status_code = 200 if orchestrator_healthy else 503
        return JSONResponse(content=health_status, status_code=status_code)
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            content={
                "service": "tasks_api",
                "healthy": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            },
            status_code=503
        )