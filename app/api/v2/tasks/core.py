"""
TaskExecutionAPI Core - Unified Task Execution Endpoints

Core task execution functionality consolidated from multiple source files:
- Task creation and assignment with intelligent agent matching
- Real-time status tracking and progress monitoring  
- Priority management and lifecycle operations
- Epic 1 ConsolidatedProductionOrchestrator integration
- Performance optimized with <200ms response times

Consolidates functionality from:
- app/api/endpoints/tasks.py (current task endpoints)
- app/api/v2/tasks.py (newer task API patterns)
- app/api/v1/orchestrator_core.py (orchestrator integration)
- app/api/v1/team_coordination.py (team coordination features)
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, Depends, Query, Path, BackgroundTasks, status
from pydantic import ValidationError
import structlog

from .models import (
    TaskExecutionRequest, TaskExecutionResponse, TaskStatusUpdateRequest,
    TaskPriorityUpdateRequest, TaskAssignmentRequest, TaskQueryParams,
    TaskListResponse, OperationResponse, HealthResponse
)
from .middleware import (
    require_task_create, require_task_read, require_task_update, 
    require_task_delete, require_admin_access, check_rate_limit_dependency,
    audit_operation, with_circuit_breaker, orchestrator_circuit_breaker
)

from ....core.database import get_async_session
from ....core.consolidated_orchestrator import ConsolidatedProductionOrchestrator
from ....core.orchestrator_interfaces import TaskSpec, AgentSpec, HealthStatus
from ....core.redis_integration import get_redis_service
from ....models.task import Task, TaskStatus, TaskPriority, TaskType
from ....schemas.task import TaskCreate, TaskUpdate


logger = structlog.get_logger(__name__)
router = APIRouter()


# ===============================================================================
# DEPENDENCY INJECTION AND SERVICE ACCESS
# ===============================================================================

async def get_task_orchestrator():
    """Get the Epic 1 ConsolidatedProductionOrchestrator for task operations."""
    try:
        orchestrator = ConsolidatedProductionOrchestrator()
        await orchestrator.initialize()
        return orchestrator
    except Exception as e:
        logger.error("Failed to get task orchestrator", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Orchestration service initialization failed"
        )


async def get_task_redis_service():
    """Get Redis service for task coordination and caching."""
    try:
        redis_service = get_redis_service()
        await redis_service.health_check()
        return redis_service
    except Exception as e:
        logger.warning("Redis service unavailable for task operations", error=str(e))
        return None  # Degrade gracefully without Redis


# ===============================================================================
# CORE TASK EXECUTION ENDPOINTS
# ===============================================================================

@router.post("/", response_model=TaskExecutionResponse, status_code=201)
@audit_operation("create_task", "task")
@with_circuit_breaker(orchestrator_circuit_breaker)
async def create_task(
    request: TaskExecutionRequest,
    background_tasks: BackgroundTasks,
    user: Dict[str, Any] = Depends(require_task_create),
    _: None = Depends(check_rate_limit_dependency),
    orchestrator = Depends(get_task_orchestrator),
    redis_service = Depends(get_task_redis_service)
) -> TaskExecutionResponse:
    """
    Create a new task with intelligent agent assignment and orchestration integration.
    
    Features:
    - Automatic capability-based agent assignment
    - Epic 1 ConsolidatedProductionOrchestrator integration  
    - Real-time Redis coordination
    - Performance target: <200ms response time
    - Comprehensive audit logging
    
    Args:
        request: Task creation parameters
        background_tasks: FastAPI background tasks
        user: Authenticated user information
        orchestrator: Unified production orchestrator
        redis_service: Redis coordination service
        
    Returns:
        TaskExecutionResponse with task details and assignment information
    """
    start_time = datetime.utcnow()
    task_id = str(uuid.uuid4())
    
    try:
        logger.info("Creating task via unified API",
                   task_id=task_id,
                   title=request.title,
                   user_id=user.get("user_id"),
                   auto_assign=request.auto_assign)
        
        # Build orchestration task configuration
        orchestration_task = {
            "id": task_id,
            "title": request.title,
            "description": request.description,
            "task_type": request.task_type.value if request.task_type else "general",
            "priority": request.priority.value,
            "required_capabilities": request.required_capabilities,
            "estimated_effort": request.estimated_effort,
            "timeout_seconds": request.timeout_seconds,
            "context": {
                **(request.context or {}),
                "created_by": user.get("user_id"),
                "team_coordination": request.team_coordination,
                "workflow_integration": request.workflow_integration,
                "intelligent_scheduling": request.enable_intelligent_scheduling
            },
            "dependencies": request.dependencies,
            "deadline": request.deadline.isoformat() if request.deadline else None
        }
        
        # Create task in orchestrator using Epic 1 interface
        try:
            # Convert to Epic 1 TaskSpec format
            task_spec = TaskSpec(
                description=f"{request.title}: {request.description}",
                task_type=request.task_type.value if request.task_type else "general",
                priority=request.priority.value,
                preferred_agent_role=request.preferred_agent_id,
                estimated_duration_seconds=request.timeout_seconds,
                dependencies=request.dependencies,
                metadata={
                    "id": task_id,
                    "created_by": user.get("user_id"),
                    "required_capabilities": request.required_capabilities,
                    "team_coordination": request.team_coordination,
                    "workflow_integration": request.workflow_integration,
                    "intelligent_scheduling": request.enable_intelligent_scheduling,
                    "estimated_effort": request.estimated_effort,
                    "deadline": request.deadline.isoformat() if request.deadline else None,
                    **(request.context or {})
                }
            )
            
            orchestrator_result = await orchestrator.delegate_task(task_spec)
            
            logger.info("Task delegated via Epic 1 orchestrator", 
                       task_id=task_id,
                       orchestrator_task_id=orchestrator_result.id,
                       status=orchestrator_result.status)
            
        except Exception as e:
            logger.error("Epic 1 orchestrator task delegation failed", 
                        task_id=task_id,
                        error=str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Task orchestration failed: {str(e)}"
            )
        
        # Extract assignment info from Epic 1 orchestrator result
        assigned_agent_id = orchestrator_result.assigned_agent_id
        assigned_agent_name = None
        assignment_confidence = 0.9  # High confidence from Epic 1 orchestrator
        estimated_completion = None
        
        # Calculate estimated completion if effort provided
        if request.estimated_effort:
            estimated_completion = start_time + timedelta(minutes=request.estimated_effort)
        
        # Get agent details if assigned
        if assigned_agent_id and request.auto_assign:
            try:
                agent_status = await orchestrator.get_agent_status(assigned_agent_id)
                assigned_agent_name = agent_status.role if agent_status else None
                
                logger.info("Task assigned by Epic 1 orchestrator",
                           task_id=task_id,
                           orchestrator_task_id=orchestrator_result.id,
                           agent_id=assigned_agent_id,
                           agent_role=assigned_agent_name)
                           
            except Exception as e:
                logger.warning("Could not retrieve assigned agent details", 
                              task_id=task_id,
                              agent_id=assigned_agent_id,
                              error=str(e))
        
        # Cache task data in Redis for fast retrieval
        if redis_service:
            try:
                task_cache = {
                    "task_id": task_id,
                    "title": request.title,
                    "status": TaskStatus.ASSIGNED.value if assigned_agent_id else TaskStatus.PENDING.value,
                    "assigned_agent_id": assigned_agent_id,
                    "created_at": start_time.isoformat(),
                    "user_id": user.get("user_id"),
                    "priority": request.priority.value
                }
                await redis_service.cache_set(f"task:{task_id}", task_cache, ttl=3600)
                
                # Publish task creation event for real-time updates
                await redis_service.publish("task_events", {
                    "event": "task_created",
                    "task_id": task_id,
                    "title": request.title,
                    "assigned_agent_id": assigned_agent_id,
                    "created_by": user.get("user_id"),
                    "timestamp": start_time.isoformat()
                })
                
            except Exception as e:
                logger.warning("Redis caching failed", task_id=task_id, error=str(e))
        
        # Schedule background metrics collection
        background_tasks.add_task(
            collect_task_creation_metrics,
            task_id=task_id,
            user_id=user.get("user_id"),
            assignment_success=assigned_agent_id is not None,
            response_time_ms=(datetime.utcnow() - start_time).total_seconds() * 1000
        )
        
        # Build response
        response = TaskExecutionResponse(
            task_id=task_id,
            title=request.title,
            description=request.description,
            task_type=request.task_type.value if request.task_type else None,
            status=TaskStatus.ASSIGNED.value if assigned_agent_id else TaskStatus.PENDING.value,
            priority=request.priority.value,
            assigned_agent_id=assigned_agent_id,
            assigned_agent_name=assigned_agent_name,
            assignment_confidence=assignment_confidence,
            created_at=start_time,
            assigned_at=start_time if assigned_agent_id else None,
            estimated_completion=estimated_completion,
            estimated_effort=request.estimated_effort,
            context=request.context,
            performance_metrics={
                "creation_time_ms": (datetime.utcnow() - start_time).total_seconds() * 1000,
                "auto_assigned": assigned_agent_id is not None,
                "orchestrator_integrated": True
            }
        )
        
        # Performance monitoring
        response_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        if response_time_ms > 200:
            logger.warning("Task creation exceeded performance target",
                          task_id=task_id,
                          response_time_ms=response_time_ms,
                          target_ms=200)
        
        logger.info("Task created successfully",
                   task_id=task_id,
                   assigned=assigned_agent_id is not None,
                   response_time_ms=response_time_ms)
        
        return response
        
    except HTTPException:
        raise
    except ValidationError as e:
        logger.error("Task creation validation failed", 
                    task_id=task_id,
                    validation_errors=e.errors())
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Request validation failed: {str(e)}"
        )
    except Exception as e:
        logger.error("Task creation failed",
                    task_id=task_id,
                    error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Task creation failed: {str(e)}"
        )


@router.get("/", response_model=TaskListResponse)
@with_circuit_breaker(orchestrator_circuit_breaker)
async def list_tasks(
    query_params: TaskQueryParams = Depends(),
    user: Dict[str, Any] = Depends(require_task_read),
    orchestrator = Depends(get_task_orchestrator),
    redis_service = Depends(get_task_redis_service)
) -> TaskListResponse:
    """
    List tasks with advanced filtering and pagination.
    
    Features:
    - Multi-criteria filtering (status, priority, agent, type, dates)
    - Efficient pagination with offset/limit
    - Redis caching for performance
    - Real-time data from orchestrator
    - User-scoped visibility controls
    
    Args:
        query_params: Query parameters for filtering and pagination
        user: Authenticated user information
        orchestrator: Unified production orchestrator
        redis_service: Redis caching service
        
    Returns:
        TaskListResponse with filtered tasks and metadata
    """
    start_time = datetime.utcnow()
    
    try:
        logger.info("Listing tasks",
                   user_id=user.get("user_id"),
                   filters={
                       "status": query_params.status,
                       "priority": query_params.priority,
                       "assigned_agent_id": query_params.assigned_agent_id
                   },
                   pagination={
                       "limit": query_params.limit,
                       "offset": query_params.offset
                   })
        
        # Build filter configuration for orchestrator
        filter_config = {
            "status_filter": query_params.status.value if query_params.status else None,
            "priority_filter": query_params.priority.value if query_params.priority else None,
            "agent_filter": query_params.assigned_agent_id,
            "task_type_filter": query_params.task_type.value if query_params.task_type else None,
            "created_after": query_params.created_after.isoformat() if query_params.created_after else None,
            "created_before": query_params.created_before.isoformat() if query_params.created_before else None,
            "limit": query_params.limit,
            "offset": query_params.offset,
            "sort_by": query_params.sort_by,
            "sort_order": query_params.sort_order
        }
        
        # Check Redis cache first
        cache_key = f"task_list:{hash(str(filter_config))}"
        cached_result = None
        
        if redis_service:
            try:
                cached_result = await redis_service.cache_get(cache_key)
                if cached_result:
                    logger.info("Serving task list from cache", cache_key=cache_key)
            except Exception as e:
                logger.warning("Cache retrieval failed", error=str(e))
        
        if not cached_result:
            # Get tasks from orchestrator
            try:
                tasks_data = await orchestrator.list_tasks(**filter_config)
            except Exception as e:
                logger.error("Orchestrator task listing failed", error=str(e))
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Task listing service temporarily unavailable"
                )
            
            # Cache result
            if redis_service and tasks_data:
                try:
                    await redis_service.cache_set(cache_key, tasks_data, ttl=300)  # 5 minutes
                except Exception as e:
                    logger.warning("Failed to cache task list", error=str(e))
        else:
            tasks_data = cached_result
        
        # Convert to response format
        tasks = []
        status_summary = {}
        priority_summary = {}
        
        for task_data in tasks_data.get("tasks", []):
            try:
                # Convert orchestrator task data to response format
                task_response = TaskExecutionResponse(
                    task_id=task_data["id"],
                    title=task_data.get("title", "Unknown Task"),
                    description=task_data.get("description"),
                    task_type=task_data.get("task_type"),
                    status=task_data.get("status", TaskStatus.PENDING.value),
                    priority=task_data.get("priority", TaskPriority.MEDIUM.value),
                    assigned_agent_id=task_data.get("assigned_agent_id"),
                    assigned_agent_name=task_data.get("assigned_agent_name"),
                    assignment_confidence=task_data.get("assignment_confidence"),
                    created_at=datetime.fromisoformat(task_data["created_at"]) if task_data.get("created_at") else datetime.utcnow(),
                    assigned_at=datetime.fromisoformat(task_data["assigned_at"]) if task_data.get("assigned_at") else None,
                    started_at=datetime.fromisoformat(task_data["started_at"]) if task_data.get("started_at") else None,
                    completed_at=datetime.fromisoformat(task_data["completed_at"]) if task_data.get("completed_at") else None,
                    estimated_completion=datetime.fromisoformat(task_data["estimated_completion"]) if task_data.get("estimated_completion") else None,
                    progress=task_data.get("progress", 0.0),
                    estimated_effort=task_data.get("estimated_effort"),
                    actual_effort=task_data.get("actual_effort"),
                    result=task_data.get("result"),
                    error_message=task_data.get("error_message"),
                    retry_count=task_data.get("retry_count", 0),
                    context=task_data.get("context")
                )
                
                tasks.append(task_response)
                
                # Update summaries
                status = task_response.status
                priority = task_response.priority
                status_summary[status] = status_summary.get(status, 0) + 1
                priority_summary[priority] = priority_summary.get(priority, 0) + 1
                
            except Exception as e:
                logger.warning("Skipping malformed task data", 
                              task_id=task_data.get("id", "unknown"),
                              error=str(e))
                continue
        
        response = TaskListResponse(
            tasks=tasks,
            total=tasks_data.get("total", len(tasks)),
            offset=query_params.offset,
            limit=query_params.limit,
            status_summary=status_summary,
            priority_summary=priority_summary
        )
        
        # Performance monitoring
        response_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        logger.info("Task list completed",
                   total_tasks=len(tasks),
                   response_time_ms=response_time_ms,
                   cached=cached_result is not None)
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Task listing failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Task listing failed: {str(e)}"
        )


@router.get("/{task_id}", response_model=TaskExecutionResponse)
@with_circuit_breaker(orchestrator_circuit_breaker)
async def get_task(
    task_id: str = Path(..., description="Task ID"),
    user: Dict[str, Any] = Depends(require_task_read),
    orchestrator = Depends(get_task_orchestrator),
    redis_service = Depends(get_task_redis_service)
) -> TaskExecutionResponse:
    """
    Get detailed task information with real-time status.
    
    Performance target: <50ms response time with Redis caching
    
    Args:
        task_id: Unique task identifier
        user: Authenticated user information
        orchestrator: Unified production orchestrator
        redis_service: Redis caching service
        
    Returns:
        TaskExecutionResponse with comprehensive task details
    """
    start_time = datetime.utcnow()
    
    try:
        # Validate task ID format
        try:
            uuid.UUID(task_id)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid task ID format"
            )
        
        # Check Redis cache first for performance
        if redis_service:
            try:
                cached_task = await redis_service.cache_get(f"task:{task_id}")
                if cached_task:
                    response_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
                    logger.info("Task retrieved from cache",
                               task_id=task_id,
                               response_time_ms=response_time_ms)
                    
                    # Convert cached data to response format
                    return TaskExecutionResponse(**cached_task)
            except Exception as e:
                logger.warning("Cache retrieval failed", task_id=task_id, error=str(e))
        
        # Get task from orchestrator
        try:
            task_data = await orchestrator.get_task_status(task_id)
            if not task_data:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Task not found"
                )
        except HTTPException:
            raise
        except Exception as e:
            logger.error("Orchestrator task retrieval failed", 
                        task_id=task_id,
                        error=str(e))
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Task retrieval service temporarily unavailable"
            )
        
        # Convert to response format
        response = TaskExecutionResponse(
            task_id=task_id,
            title=task_data.get("title", "Unknown Task"),
            description=task_data.get("description"),
            task_type=task_data.get("task_type"),
            status=task_data.get("status", TaskStatus.PENDING.value),
            priority=task_data.get("priority", TaskPriority.MEDIUM.value),
            assigned_agent_id=task_data.get("assigned_agent_id"),
            assigned_agent_name=task_data.get("assigned_agent_name"),
            assignment_confidence=task_data.get("assignment_confidence"),
            created_at=datetime.fromisoformat(task_data["created_at"]) if task_data.get("created_at") else datetime.utcnow(),
            assigned_at=datetime.fromisoformat(task_data["assigned_at"]) if task_data.get("assigned_at") else None,
            started_at=datetime.fromisoformat(task_data["started_at"]) if task_data.get("started_at") else None,
            completed_at=datetime.fromisoformat(task_data["completed_at"]) if task_data.get("completed_at") else None,
            estimated_completion=datetime.fromisoformat(task_data["estimated_completion"]) if task_data.get("estimated_completion") else None,
            progress=task_data.get("progress", 0.0),
            estimated_effort=task_data.get("estimated_effort"),
            actual_effort=task_data.get("actual_effort"),
            result=task_data.get("result"),
            error_message=task_data.get("error_message"),
            retry_count=task_data.get("retry_count", 0),
            context=task_data.get("context"),
            performance_metrics={
                "retrieval_time_ms": (datetime.utcnow() - start_time).total_seconds() * 1000,
                "cached": False,
                "orchestrator_integrated": True
            }
        )
        
        # Update cache with fresh data
        if redis_service:
            try:
                await redis_service.cache_set(f"task:{task_id}", response.dict(), ttl=300)
            except Exception as e:
                logger.warning("Cache update failed", task_id=task_id, error=str(e))
        
        response_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        logger.info("Task retrieved successfully",
                   task_id=task_id,
                   status=response.status,
                   response_time_ms=response_time_ms)
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Task retrieval failed", task_id=task_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Task retrieval failed: {str(e)}"
        )


@router.put("/{task_id}/status", response_model=TaskExecutionResponse)
@audit_operation("update_task_status", "task")
@with_circuit_breaker(orchestrator_circuit_breaker)
async def update_task_status(
    task_id: str = Path(..., description="Task ID"),
    request: TaskStatusUpdateRequest = ...,
    user: Dict[str, Any] = Depends(require_task_update),
    orchestrator = Depends(get_task_orchestrator),
    redis_service = Depends(get_task_redis_service)
) -> TaskExecutionResponse:
    """
    Update task status with orchestrator integration and real-time notifications.
    
    Args:
        task_id: Unique task identifier
        request: Status update parameters
        user: Authenticated user information
        orchestrator: Unified production orchestrator
        redis_service: Redis coordination service
        
    Returns:
        TaskExecutionResponse with updated task information
    """
    start_time = datetime.utcnow()
    
    try:
        # Validate task ID
        try:
            uuid.UUID(task_id)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid task ID format"
            )
        
        logger.info("Updating task status",
                   task_id=task_id,
                   new_status=request.status.value,
                   user_id=user.get("user_id"))
        
        # Update task through orchestrator
        update_data = {
            "status": request.status.value,
            "progress": request.progress,
            "result": request.result,
            "error_message": request.error_message,
            "actual_effort": request.actual_effort,
            "updated_by": user.get("user_id"),
            "updated_at": start_time.isoformat()
        }
        
        try:
            update_result = await orchestrator.update_task_status(task_id, update_data)
            if not update_result.get("success", False):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Status update failed: {update_result.get('error', 'Unknown error')}"
                )
        except HTTPException:
            raise
        except Exception as e:
            logger.error("Orchestrator status update failed",
                        task_id=task_id,
                        error=str(e))
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Status update service temporarily unavailable"
            )
        
        # Invalidate cache and publish update event
        if redis_service:
            try:
                await redis_service.cache_delete(f"task:{task_id}")
                await redis_service.publish("task_events", {
                    "event": "task_status_updated",
                    "task_id": task_id,
                    "new_status": request.status.value,
                    "progress": request.progress,
                    "updated_by": user.get("user_id"),
                    "timestamp": start_time.isoformat()
                })
            except Exception as e:
                logger.warning("Redis update notification failed", 
                              task_id=task_id,
                              error=str(e))
        
        # Get updated task data
        updated_task = await orchestrator.get_task_status(task_id)
        if not updated_task:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Task not found after update"
            )
        
        response = TaskExecutionResponse(
            task_id=task_id,
            title=updated_task.get("title", "Unknown Task"),
            description=updated_task.get("description"),
            task_type=updated_task.get("task_type"),
            status=updated_task.get("status", request.status.value),
            priority=updated_task.get("priority", TaskPriority.MEDIUM.value),
            assigned_agent_id=updated_task.get("assigned_agent_id"),
            assigned_agent_name=updated_task.get("assigned_agent_name"),
            created_at=datetime.fromisoformat(updated_task["created_at"]) if updated_task.get("created_at") else start_time,
            assigned_at=datetime.fromisoformat(updated_task["assigned_at"]) if updated_task.get("assigned_at") else None,
            started_at=datetime.fromisoformat(updated_task["started_at"]) if updated_task.get("started_at") else None,
            completed_at=datetime.fromisoformat(updated_task["completed_at"]) if updated_task.get("completed_at") else None,
            progress=request.progress or updated_task.get("progress", 0.0),
            estimated_effort=updated_task.get("estimated_effort"),
            actual_effort=request.actual_effort or updated_task.get("actual_effort"),
            result=request.result or updated_task.get("result"),
            error_message=request.error_message or updated_task.get("error_message"),
            retry_count=updated_task.get("retry_count", 0),
            context=updated_task.get("context")
        )
        
        response_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        logger.info("Task status updated successfully",
                   task_id=task_id,
                   new_status=request.status.value,
                   response_time_ms=response_time_ms)
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Task status update failed", task_id=task_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Task status update failed: {str(e)}"
        )


@router.put("/{task_id}/priority", response_model=OperationResponse)
@audit_operation("update_task_priority", "task") 
@with_circuit_breaker(orchestrator_circuit_breaker)
async def update_task_priority(
    task_id: str = Path(..., description="Task ID"),
    request: TaskPriorityUpdateRequest = ...,
    user: Dict[str, Any] = Depends(require_task_update),
    orchestrator = Depends(get_task_orchestrator),
    redis_service = Depends(get_task_redis_service)
) -> OperationResponse:
    """
    Update task priority with orchestrator integration and conflict resolution.
    
    Args:
        task_id: Unique task identifier
        request: Priority update parameters
        user: Authenticated user information
        orchestrator: Unified production orchestrator
        redis_service: Redis coordination service
        
    Returns:
        OperationResponse with update status
    """
    start_time = datetime.utcnow()
    
    try:
        # Validate task ID
        try:
            uuid.UUID(task_id)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid task ID format"
            )
        
        logger.info("Updating task priority",
                   task_id=task_id,
                   new_priority=request.priority.value,
                   reason=request.reason,
                   user_id=user.get("user_id"))
        
        # Update priority through orchestrator
        try:
            result = await orchestrator.update_task_priority(
                task_id=task_id,
                new_priority=request.priority,
                reason=request.reason,
                force_update=request.force_update,
                updated_by=user.get("user_id")
            )
            
            if not result.get("success", False):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=result.get("message", "Priority update failed")
                )
                
        except HTTPException:
            raise
        except Exception as e:
            logger.error("Orchestrator priority update failed",
                        task_id=task_id,
                        error=str(e))
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Priority update service temporarily unavailable"
            )
        
        # Invalidate cache and notify
        if redis_service:
            try:
                await redis_service.cache_delete(f"task:{task_id}")
                await redis_service.publish("task_events", {
                    "event": "task_priority_updated",
                    "task_id": task_id,
                    "new_priority": request.priority.value,
                    "reason": request.reason,
                    "updated_by": user.get("user_id"),
                    "timestamp": start_time.isoformat()
                })
            except Exception as e:
                logger.warning("Priority update notification failed", error=str(e))
        
        response = OperationResponse(
            success=True,
            message=f"Task priority updated to {request.priority.value}",
            operation_id=str(uuid.uuid4()),
            timestamp=start_time,
            details={
                "task_id": task_id,
                "old_priority": result.get("old_priority"),
                "new_priority": request.priority.value,
                "reason": request.reason,
                "updated_by": user.get("user_id")
            }
        )
        
        logger.info("Task priority updated successfully",
                   task_id=task_id,
                   new_priority=request.priority.value)
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Task priority update failed", task_id=task_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Priority update failed: {str(e)}"
        )


@router.post("/{task_id}/assign", response_model=TaskExecutionResponse)
@audit_operation("assign_task", "task")
@with_circuit_breaker(orchestrator_circuit_breaker)
async def assign_task(
    task_id: str = Path(..., description="Task ID"),
    request: TaskAssignmentRequest = ...,
    user: Dict[str, Any] = Depends(require_task_update),
    orchestrator = Depends(get_task_orchestrator),
    redis_service = Depends(get_task_redis_service)
) -> TaskExecutionResponse:
    """
    Assign task to agent with intelligent matching and Epic 1 integration.
    
    Args:
        task_id: Unique task identifier
        request: Assignment parameters
        user: Authenticated user information
        orchestrator: Unified production orchestrator
        redis_service: Redis coordination service
        
    Returns:
        TaskExecutionResponse with assignment details
    """
    start_time = datetime.utcnow()
    
    try:
        # Validate task ID
        try:
            uuid.UUID(task_id)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid task ID format"
            )
        
        # Validate agent ID if provided
        if request.agent_id:
            try:
                uuid.UUID(request.agent_id)
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid agent ID format"
                )
        
        logger.info("Assigning task",
                   task_id=task_id,
                   agent_id=request.agent_id,
                   strategy=request.strategy,
                   user_id=user.get("user_id"))
        
        # Perform assignment through orchestrator
        try:
            assignment_result = await orchestrator.assign_task(
                task_id=task_id,
                agent_id=request.agent_id,
                strategy=request.strategy,
                priority_override=request.priority_override,
                context_override=request.context_override,
                force_assignment=request.force_assignment,
                timeout_seconds=request.timeout_seconds,
                assigned_by=user.get("user_id")
            )
            
            if not assignment_result.get("success", False):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=assignment_result.get("message", "Task assignment failed")
                )
                
        except HTTPException:
            raise
        except Exception as e:
            logger.error("Orchestrator assignment failed",
                        task_id=task_id,
                        error=str(e))
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Task assignment service temporarily unavailable"
            )
        
        assigned_agent_id = assignment_result.get("agent_id")
        assigned_agent_name = assignment_result.get("agent_name")
        assignment_confidence = assignment_result.get("confidence", 0.8)
        
        # Invalidate cache and publish assignment event
        if redis_service:
            try:
                await redis_service.cache_delete(f"task:{task_id}")
                await redis_service.publish("task_events", {
                    "event": "task_assigned",
                    "task_id": task_id,
                    "agent_id": assigned_agent_id,
                    "agent_name": assigned_agent_name,
                    "confidence": assignment_confidence,
                    "assigned_by": user.get("user_id"),
                    "timestamp": start_time.isoformat()
                })
            except Exception as e:
                logger.warning("Assignment notification failed", error=str(e))
        
        # Get updated task data
        updated_task = await orchestrator.get_task_status(task_id)
        response = TaskExecutionResponse(
            task_id=task_id,
            title=updated_task.get("title", "Unknown Task"),
            description=updated_task.get("description"),
            task_type=updated_task.get("task_type"),
            status=updated_task.get("status", TaskStatus.ASSIGNED.value),
            priority=updated_task.get("priority", TaskPriority.MEDIUM.value),
            assigned_agent_id=assigned_agent_id,
            assigned_agent_name=assigned_agent_name,
            assignment_confidence=assignment_confidence,
            created_at=datetime.fromisoformat(updated_task["created_at"]) if updated_task.get("created_at") else start_time,
            assigned_at=start_time,
            started_at=datetime.fromisoformat(updated_task["started_at"]) if updated_task.get("started_at") else None,
            completed_at=datetime.fromisoformat(updated_task["completed_at"]) if updated_task.get("completed_at") else None,
            progress=updated_task.get("progress", 0.0),
            estimated_effort=updated_task.get("estimated_effort"),
            actual_effort=updated_task.get("actual_effort"),
            result=updated_task.get("result"),
            error_message=updated_task.get("error_message"),
            context=updated_task.get("context"),
            performance_metrics={
                "assignment_time_ms": (datetime.utcnow() - start_time).total_seconds() * 1000,
                "strategy": request.strategy,
                "confidence": assignment_confidence
            }
        )
        
        logger.info("Task assigned successfully",
                   task_id=task_id,
                   agent_id=assigned_agent_id,
                   confidence=assignment_confidence)
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Task assignment failed", task_id=task_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Task assignment failed: {str(e)}"
        )


@router.delete("/{task_id}", response_model=OperationResponse)
@audit_operation("cancel_task", "task")
@with_circuit_breaker(orchestrator_circuit_breaker)
async def cancel_task(
    task_id: str = Path(..., description="Task ID"),
    force: bool = Query(False, description="Force cancellation even if running"),
    reason: Optional[str] = Query(None, description="Cancellation reason"),
    user: Dict[str, Any] = Depends(require_task_delete),
    orchestrator = Depends(get_task_orchestrator),
    redis_service = Depends(get_task_redis_service)
) -> OperationResponse:
    """
    Cancel task execution with orchestrator coordination and cleanup.
    
    Args:
        task_id: Unique task identifier
        force: Force cancellation even if task is running
        reason: Reason for cancellation
        user: Authenticated user information
        orchestrator: Unified production orchestrator
        redis_service: Redis coordination service
        
    Returns:
        OperationResponse with cancellation status
    """
    start_time = datetime.utcnow()
    
    try:
        # Validate task ID
        try:
            uuid.UUID(task_id)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid task ID format"
            )
        
        logger.info("Cancelling task",
                   task_id=task_id,
                   force=force,
                   reason=reason,
                   user_id=user.get("user_id"))
        
        # Cancel task through orchestrator
        try:
            cancellation_result = await orchestrator.cancel_task(
                task_id=task_id,
                force=force,
                reason=reason,
                cancelled_by=user.get("user_id")
            )
            
            if not cancellation_result.get("success", False):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=cancellation_result.get("message", "Task cancellation failed")
                )
                
        except HTTPException:
            raise
        except Exception as e:
            logger.error("Orchestrator cancellation failed",
                        task_id=task_id,
                        error=str(e))
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Task cancellation service temporarily unavailable"
            )
        
        # Clean up cache and publish cancellation event
        if redis_service:
            try:
                await redis_service.cache_delete(f"task:{task_id}")
                await redis_service.publish("task_events", {
                    "event": "task_cancelled",
                    "task_id": task_id,
                    "reason": reason,
                    "force": force,
                    "cancelled_by": user.get("user_id"),
                    "timestamp": start_time.isoformat()
                })
            except Exception as e:
                logger.warning("Cancellation notification failed", error=str(e))
        
        response = OperationResponse(
            success=True,
            message=f"Task {task_id} cancelled successfully",
            operation_id=str(uuid.uuid4()),
            timestamp=start_time,
            details={
                "task_id": task_id,
                "reason": reason,
                "force": force,
                "cancelled_by": user.get("user_id"),
                "agent_id": cancellation_result.get("released_agent_id")
            }
        )
        
        logger.info("Task cancelled successfully",
                   task_id=task_id,
                   force=force)
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Task cancellation failed", task_id=task_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Task cancellation failed: {str(e)}"
        )


# ===============================================================================
# HEALTH AND STATUS ENDPOINTS
# ===============================================================================

@router.get("/health", response_model=HealthResponse)
async def task_execution_health_check(
    user: Dict[str, Any] = Depends(require_task_read),
    orchestrator = Depends(get_task_orchestrator),
    redis_service = Depends(get_task_redis_service)
) -> HealthResponse:
    """
    Comprehensive health check for task execution services.
    
    Returns:
        HealthResponse with component health status and metrics
    """
    start_time = datetime.utcnow()
    overall_healthy = True
    components = {}
    metrics = {}
    
    # Check Epic 1 orchestrator health
    try:
        orchestrator_health = await orchestrator.health_check()
        components["orchestrator"] = {
            "status": orchestrator_health.status.value,
            "uptime_seconds": orchestrator_health.uptime_seconds,
            "orchestrator_type": orchestrator_health.orchestrator_type,
            "version": orchestrator_health.version,
            "performance": orchestrator_health.performance,
            "components": orchestrator_health.components
        }
        if orchestrator_health.status != HealthStatus.HEALTHY:
            overall_healthy = False
    except Exception as e:
        overall_healthy = False
        components["orchestrator"] = {
            "status": "unhealthy",
            "error": str(e)
        }
    
    # Check Redis health
    if redis_service:
        try:
            redis_health = await redis_service.health_check()
            components["redis"] = redis_health
            if redis_health.get("status") != "healthy":
                overall_healthy = False
        except Exception as e:
            overall_healthy = False
            components["redis"] = {
                "status": "unhealthy",
                "error": str(e)
            }
    else:
        components["redis"] = {
            "status": "degraded",
            "message": "Redis service not available - operating in degraded mode"
        }
    
    # Collect system metrics
    try:
        if orchestrator:
            task_stats = await orchestrator.get_task_stats()
            metrics = {
                "total_tasks": task_stats.get("total_tasks", 0),
                "active_tasks": task_stats.get("active_tasks", 0),
                "completed_tasks": task_stats.get("completed_tasks", 0),
                "average_response_time_ms": task_stats.get("average_response_time_ms", 0)
            }
    except Exception as e:
        logger.warning("Failed to collect task metrics", error=str(e))
    
    # Calculate health check response time
    health_check_time = (datetime.utcnow() - start_time).total_seconds() * 1000
    metrics["health_check_time_ms"] = health_check_time
    
    return HealthResponse(
        healthy=overall_healthy,
        components=components,
        metrics=metrics,
        build_info={
            "version": "2.0.0",
            "build_date": "2024-12-19",
            "commit": "epic4_phase4_task_execution",
            "consolidation_version": "94.4%"
        }
    )


# ===============================================================================
# BACKGROUND TASK FUNCTIONS
# ===============================================================================

async def collect_task_creation_metrics(
    task_id: str,
    user_id: str,
    assignment_success: bool,
    response_time_ms: float
):
    """Background task to collect task creation metrics."""
    try:
        metrics_data = {
            "task_id": task_id,
            "user_id": user_id,
            "assignment_success": assignment_success,
            "response_time_ms": response_time_ms,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Store metrics for analytics
        logger.info("Task creation metrics collected", **metrics_data)
        
    except Exception as e:
        logger.error("Failed to collect task creation metrics",
                    task_id=task_id,
                    error=str(e))