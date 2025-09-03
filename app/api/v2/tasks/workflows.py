"""
TaskExecutionAPI Workflows - Unified Workflow Orchestration

Consolidated workflow orchestration functionality from:
- app/api/v1/workflows.py (comprehensive workflow management)
- app/api/v1/orchestrator_core.py (orchestration patterns)
- app/api/v1/team_coordination.py (team workflow coordination)

Features:
- Multi-agent workflow coordination with dependency management
- Real-time state management and progress tracking
- Epic 1 ConsolidatedProductionOrchestrator integration
- Error handling and recovery mechanisms  
- Performance optimized with intelligent caching
- Team coordination with WebSocket real-time updates
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, Depends, Query, Path, BackgroundTasks, status
import structlog

from .models import (
    WorkflowExecutionRequest, WorkflowExecutionResponse, WorkflowTaskAssignmentRequest,
    TaskQueryParams, OperationResponse
)
from .middleware import (
    require_workflow_create, require_workflow_execute, require_task_read,
    check_rate_limit_dependency, audit_operation, with_circuit_breaker,
    workflow_circuit_breaker
)

from app.core.production_orchestrator import create_production_orchestrator
from app.core.redis_integration import get_redis_service
from app.models.workflow import Workflow, WorkflowStatus, WorkflowPriority
from app.models.task import Task, TaskStatus


logger = structlog.get_logger(__name__)
router = APIRouter()


# ===============================================================================
# WORKFLOW ORCHESTRATION SERVICE
# ===============================================================================

class WorkflowOrchestrationService:
    """
    Unified workflow orchestration service consolidating multiple source patterns.
    
    Provides comprehensive workflow management with Epic 1 integration,
    team coordination, and intelligent state management.
    """
    
    def __init__(self):
        self.orchestrator = None
        self.redis_service = None
        
    async def initialize(self):
        """Initialize service dependencies."""
        self.orchestrator = create_production_orchestrator()
        self.redis_service = get_redis_service()
    
    async def create_workflow_execution(
        self,
        request: WorkflowExecutionRequest,
        user_id: str
    ) -> Dict[str, Any]:
        """
        Create and initialize a workflow execution with comprehensive orchestration.
        
        Args:
            request: Workflow execution parameters
            user_id: User creating the workflow
            
        Returns:
            Dict containing workflow execution details
        """
        workflow_id = str(uuid.uuid4())
        creation_time = datetime.utcnow()
        
        try:
            # Build workflow configuration for orchestrator
            workflow_config = {
                "id": workflow_id,
                "name": request.name,
                "description": request.description,
                "priority": request.priority.value,
                "definition": request.definition,
                "task_ids": request.task_ids,
                "dependencies": request.dependencies or {},
                "context": {
                    **(request.context or {}),
                    "created_by": user_id,
                    "parallel_execution": request.parallel_execution,
                    "intelligent_scheduling": request.intelligent_scheduling,
                    "team_coordination": request.team_coordination,
                    "auto_recovery": request.auto_recovery
                },
                "variables": request.variables or {},
                "estimated_duration": request.estimated_duration,
                "due_date": request.due_date.isoformat() if request.due_date else None,
                "created_at": creation_time.isoformat()
            }
            
            # Create workflow in orchestrator with Epic 1 integration
            orchestrator_result = await self.orchestrator.create_workflow(workflow_config)
            if not orchestrator_result.get("success", False):
                raise Exception(f"Orchestrator workflow creation failed: {orchestrator_result.get('error')}")
            
            # Initialize workflow state management
            workflow_state = {
                "workflow_id": workflow_id,
                "status": WorkflowStatus.CREATED.value,
                "total_tasks": len(request.task_ids),
                "completed_tasks": 0,
                "failed_tasks": 0,
                "pending_tasks": len(request.task_ids),
                "current_tasks": [],
                "ready_tasks": [],
                "blocked_tasks": [],
                "completion_percentage": 0.0,
                "created_at": creation_time.isoformat(),
                "updated_at": creation_time.isoformat()
            }
            
            # Cache workflow state in Redis for fast access
            if self.redis_service:
                await self.redis_service.cache_set(
                    f"workflow:{workflow_id}",
                    workflow_state,
                    ttl=86400  # 24 hours
                )
                
                # Publish workflow creation event
                await self.redis_service.publish("workflow_events", {
                    "event": "workflow_created",
                    "workflow_id": workflow_id,
                    "name": request.name,
                    "total_tasks": len(request.task_ids),
                    "created_by": user_id,
                    "timestamp": creation_time.isoformat()
                })
            
            logger.info("Workflow execution created",
                       workflow_id=workflow_id,
                       name=request.name,
                       total_tasks=len(request.task_ids))
            
            return {
                "success": True,
                "workflow_id": workflow_id,
                "orchestrator_result": orchestrator_result,
                "state": workflow_state
            }
            
        except Exception as e:
            logger.error("Workflow creation failed",
                        workflow_id=workflow_id,
                        error=str(e))
            raise
    
    async def execute_workflow_with_coordination(
        self,
        workflow_id: str,
        execution_context: Optional[Dict[str, Any]] = None,
        user_id: str = None
    ) -> Dict[str, Any]:
        """
        Execute workflow with comprehensive coordination and state management.
        
        Args:
            workflow_id: Unique workflow identifier
            execution_context: Execution context override
            user_id: User executing the workflow
            
        Returns:
            Dict containing execution results and state
        """
        start_time = datetime.utcnow()
        
        try:
            # Get current workflow state
            workflow_state = None
            if self.redis_service:
                workflow_state = await self.redis_service.cache_get(f"workflow:{workflow_id}")
            
            if not workflow_state:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Workflow not found or state unavailable"
                )
            
            # Validate workflow can be executed
            current_status = workflow_state.get("status")
            if current_status not in [WorkflowStatus.CREATED.value, WorkflowStatus.READY.value, WorkflowStatus.PAUSED.value]:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Workflow cannot be executed in status: {current_status}"
                )
            
            # Update workflow state to running
            workflow_state["status"] = WorkflowStatus.RUNNING.value
            workflow_state["started_at"] = start_time.isoformat()
            workflow_state["updated_at"] = start_time.isoformat()
            
            if execution_context:
                workflow_state["execution_context"] = execution_context
            
            # Execute workflow through orchestrator
            execution_result = await self.orchestrator.execute_workflow(
                workflow_id=workflow_id,
                context_override=execution_context,
                executed_by=user_id
            )
            
            if not execution_result.get("success", False):
                workflow_state["status"] = WorkflowStatus.FAILED.value
                workflow_state["error"] = execution_result.get("error", "Execution failed")
                
                # Update cache with failed state
                if self.redis_service:
                    await self.redis_service.cache_set(f"workflow:{workflow_id}", workflow_state, ttl=86400)
                
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Workflow execution failed: {execution_result.get('error')}"
                )
            
            # Initialize task execution tracking
            await self._initialize_task_execution_tracking(workflow_id, workflow_state)
            
            # Update cache with running state
            if self.redis_service:
                await self.redis_service.cache_set(f"workflow:{workflow_id}", workflow_state, ttl=86400)
                
                # Publish execution started event
                await self.redis_service.publish("workflow_events", {
                    "event": "workflow_execution_started",
                    "workflow_id": workflow_id,
                    "started_by": user_id,
                    "timestamp": start_time.isoformat()
                })
            
            logger.info("Workflow execution started",
                       workflow_id=workflow_id,
                       user_id=user_id)
            
            return {
                "success": True,
                "execution_id": execution_result.get("execution_id"),
                "state": workflow_state,
                "estimated_completion": self._calculate_estimated_completion(workflow_state)
            }
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error("Workflow execution failed",
                        workflow_id=workflow_id,
                        error=str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Workflow execution failed: {str(e)}"
            )
    
    async def get_workflow_progress_detailed(
        self,
        workflow_id: str
    ) -> Dict[str, Any]:
        """
        Get comprehensive workflow progress with real-time state tracking.
        
        Args:
            workflow_id: Unique workflow identifier
            
        Returns:
            Dict containing detailed progress information
        """
        try:
            # Get workflow state from cache first
            workflow_state = None
            if self.redis_service:
                workflow_state = await self.redis_service.cache_get(f"workflow:{workflow_id}")
            
            # Fall back to orchestrator if cache miss
            if not workflow_state:
                orchestrator_state = await self.orchestrator.get_workflow_state(workflow_id)
                if not orchestrator_state:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail="Workflow not found"
                    )
                workflow_state = orchestrator_state
            
            # Get real-time task states
            task_states = await self._get_workflow_task_states(workflow_id, workflow_state)
            
            # Update progress calculations
            progress_info = await self._calculate_workflow_progress(workflow_state, task_states)
            
            # Get performance metrics
            performance_metrics = await self._get_workflow_performance_metrics(workflow_id)
            
            return {
                "workflow_id": workflow_id,
                "name": workflow_state.get("name", "Unknown Workflow"),
                "status": workflow_state.get("status"),
                "progress": progress_info,
                "task_states": task_states,
                "performance_metrics": performance_metrics,
                "estimated_completion": self._calculate_estimated_completion(workflow_state),
                "last_updated": datetime.utcnow().isoformat()
            }
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error("Failed to get workflow progress",
                        workflow_id=workflow_id,
                        error=str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Progress tracking failed: {str(e)}"
            )
    
    async def _initialize_task_execution_tracking(
        self,
        workflow_id: str,
        workflow_state: Dict[str, Any]
    ):
        """Initialize task execution tracking for workflow."""
        try:
            task_ids = workflow_state.get("task_ids", [])
            dependencies = workflow_state.get("dependencies", {})
            
            # Determine ready tasks (no dependencies)
            ready_tasks = []
            for task_id in task_ids:
                if str(task_id) not in dependencies or not dependencies[str(task_id)]:
                    ready_tasks.append(task_id)
            
            workflow_state["ready_tasks"] = ready_tasks
            workflow_state["blocked_tasks"] = list(set(task_ids) - set(ready_tasks))
            
        except Exception as e:
            logger.error("Task tracking initialization failed",
                        workflow_id=workflow_id,
                        error=str(e))
    
    async def _get_workflow_task_states(
        self,
        workflow_id: str,
        workflow_state: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """Get current states of all workflow tasks."""
        task_states = {}
        task_ids = workflow_state.get("task_ids", [])
        
        try:
            # Get task states from orchestrator
            for task_id in task_ids:
                task_state = await self.orchestrator.get_task_status(task_id)
                if task_state:
                    task_states[task_id] = task_state
            
            return task_states
            
        except Exception as e:
            logger.warning("Failed to get some task states",
                          workflow_id=workflow_id,
                          error=str(e))
            return task_states
    
    async def _calculate_workflow_progress(
        self,
        workflow_state: Dict[str, Any],
        task_states: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate comprehensive workflow progress metrics."""
        total_tasks = workflow_state.get("total_tasks", 0)
        
        if total_tasks == 0:
            return {
                "completion_percentage": 100.0,
                "completed_tasks": 0,
                "failed_tasks": 0,
                "pending_tasks": 0,
                "in_progress_tasks": 0
            }
        
        completed_count = 0
        failed_count = 0
        in_progress_count = 0
        pending_count = 0
        
        for task_id, task_state in task_states.items():
            status = task_state.get("status", TaskStatus.PENDING.value)
            
            if status == TaskStatus.COMPLETED.value:
                completed_count += 1
            elif status == TaskStatus.FAILED.value:
                failed_count += 1
            elif status == TaskStatus.IN_PROGRESS.value:
                in_progress_count += 1
            else:
                pending_count += 1
        
        completion_percentage = (completed_count / total_tasks) * 100.0
        
        return {
            "completion_percentage": completion_percentage,
            "completed_tasks": completed_count,
            "failed_tasks": failed_count,
            "pending_tasks": pending_count,
            "in_progress_tasks": in_progress_count,
            "success_rate": (completed_count / max(completed_count + failed_count, 1)) * 100.0
        }
    
    async def _get_workflow_performance_metrics(
        self,
        workflow_id: str
    ) -> Dict[str, Any]:
        """Get workflow performance metrics."""
        try:
            # Get performance data from orchestrator
            performance_data = await self.orchestrator.get_workflow_performance_metrics(workflow_id)
            return performance_data or {}
            
        except Exception as e:
            logger.warning("Failed to get performance metrics",
                          workflow_id=workflow_id,
                          error=str(e))
            return {}
    
    def _calculate_estimated_completion(
        self,
        workflow_state: Dict[str, Any]
    ) -> Optional[str]:
        """Calculate estimated workflow completion time."""
        try:
            started_at = workflow_state.get("started_at")
            estimated_duration = workflow_state.get("estimated_duration")
            
            if started_at and estimated_duration:
                start_time = datetime.fromisoformat(started_at)
                estimated_completion = start_time + timedelta(minutes=estimated_duration)
                return estimated_completion.isoformat()
            
            return None
            
        except Exception as e:
            logger.warning("Failed to calculate estimated completion", error=str(e))
            return None


# Global service instance
workflow_service = WorkflowOrchestrationService()


# ===============================================================================
# WORKFLOW ORCHESTRATION ENDPOINTS
# ===============================================================================

@router.on_event("startup")
async def startup_workflow_service():
    """Initialize workflow orchestration service."""
    await workflow_service.initialize()


@router.post("/", response_model=WorkflowExecutionResponse, status_code=201)
@audit_operation("create_workflow", "workflow")
@with_circuit_breaker(workflow_circuit_breaker)
async def create_workflow(
    request: WorkflowExecutionRequest,
    background_tasks: BackgroundTasks,
    user: Dict[str, Any] = Depends(require_workflow_create),
    _: None = Depends(check_rate_limit_dependency)
) -> WorkflowExecutionResponse:
    """
    Create a new workflow execution with comprehensive orchestration.
    
    Features:
    - Multi-agent task coordination with dependency management
    - Epic 1 ConsolidatedProductionOrchestrator integration
    - Real-time state management with Redis caching
    - Team coordination and intelligent scheduling
    - Performance target: <500ms workflow creation
    
    Args:
        request: Workflow creation parameters
        background_tasks: FastAPI background tasks
        user: Authenticated user information
        
    Returns:
        WorkflowExecutionResponse with workflow details
    """
    start_time = datetime.utcnow()
    
    try:
        logger.info("Creating workflow execution",
                   name=request.name,
                   total_tasks=len(request.task_ids),
                   user_id=user.get("user_id"),
                   parallel_execution=request.parallel_execution)
        
        # Create workflow through service
        creation_result = await workflow_service.create_workflow_execution(
            request=request,
            user_id=user.get("user_id")
        )
        
        if not creation_result.get("success", False):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Workflow creation failed"
            )
        
        workflow_id = creation_result["workflow_id"]
        workflow_state = creation_result["state"]
        
        # Schedule background workflow monitoring
        background_tasks.add_task(
            monitor_workflow_execution,
            workflow_id=workflow_id,
            user_id=user.get("user_id")
        )
        
        # Build response
        response = WorkflowExecutionResponse(
            workflow_id=workflow_id,
            name=request.name,
            description=request.description,
            status=workflow_state.get("status", WorkflowStatus.CREATED.value),
            priority=request.priority.value,
            completion_percentage=workflow_state.get("completion_percentage", 0.0),
            total_tasks=workflow_state.get("total_tasks", 0),
            completed_tasks=workflow_state.get("completed_tasks", 0),
            failed_tasks=workflow_state.get("failed_tasks", 0),
            pending_tasks=workflow_state.get("pending_tasks", 0),
            created_at=datetime.fromisoformat(workflow_state["created_at"]),
            current_tasks=workflow_state.get("current_tasks", []),
            ready_tasks=workflow_state.get("ready_tasks", []),
            blocked_tasks=workflow_state.get("blocked_tasks", []),
            performance_metrics={
                "creation_time_ms": (datetime.utcnow() - start_time).total_seconds() * 1000,
                "orchestrator_integrated": True,
                "team_coordination_enabled": request.team_coordination,
                "intelligent_scheduling_enabled": request.intelligent_scheduling
            }
        )
        
        # Performance monitoring
        creation_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        if creation_time_ms > 500:
            logger.warning("Workflow creation exceeded performance target",
                          workflow_id=workflow_id,
                          creation_time_ms=creation_time_ms,
                          target_ms=500)
        
        logger.info("Workflow created successfully",
                   workflow_id=workflow_id,
                   total_tasks=len(request.task_ids),
                   creation_time_ms=creation_time_ms)
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Workflow creation failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Workflow creation failed: {str(e)}"
        )


@router.post("/{workflow_id}/execute", response_model=WorkflowExecutionResponse)
@audit_operation("execute_workflow", "workflow")
@with_circuit_breaker(workflow_circuit_breaker)
async def execute_workflow(
    workflow_id: str = Path(..., description="Workflow ID"),
    execution_context: Optional[Dict[str, Any]] = None,
    user: Dict[str, Any] = Depends(require_workflow_execute),
    _: None = Depends(check_rate_limit_dependency)
) -> WorkflowExecutionResponse:
    """
    Execute workflow with comprehensive coordination and state management.
    
    Features:
    - Real-time orchestration with Epic 1 integration
    - Dependency resolution and task scheduling  
    - Team coordination and parallel execution
    - Error handling and recovery mechanisms
    - Performance target: <500ms execution initiation
    
    Args:
        workflow_id: Unique workflow identifier
        execution_context: Execution context override
        user: Authenticated user information
        
    Returns:
        WorkflowExecutionResponse with execution details
    """
    start_time = datetime.utcnow()
    
    try:
        # Validate workflow ID
        try:
            uuid.UUID(workflow_id)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid workflow ID format"
            )
        
        logger.info("Executing workflow",
                   workflow_id=workflow_id,
                   user_id=user.get("user_id"))
        
        # Execute workflow through service
        execution_result = await workflow_service.execute_workflow_with_coordination(
            workflow_id=workflow_id,
            execution_context=execution_context,
            user_id=user.get("user_id")
        )
        
        if not execution_result.get("success", False):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Workflow execution failed"
            )
        
        workflow_state = execution_result["state"]
        
        # Build response
        response = WorkflowExecutionResponse(
            workflow_id=workflow_id,
            name=workflow_state.get("name", "Unknown Workflow"),
            description=workflow_state.get("description"),
            status=workflow_state.get("status", WorkflowStatus.RUNNING.value),
            priority=workflow_state.get("priority", WorkflowPriority.MEDIUM.value),
            completion_percentage=workflow_state.get("completion_percentage", 0.0),
            total_tasks=workflow_state.get("total_tasks", 0),
            completed_tasks=workflow_state.get("completed_tasks", 0),
            failed_tasks=workflow_state.get("failed_tasks", 0),
            pending_tasks=workflow_state.get("pending_tasks", 0),
            created_at=datetime.fromisoformat(workflow_state["created_at"]),
            started_at=datetime.fromisoformat(workflow_state["started_at"]),
            estimated_completion=datetime.fromisoformat(execution_result["estimated_completion"]) if execution_result.get("estimated_completion") else None,
            current_tasks=workflow_state.get("current_tasks", []),
            ready_tasks=workflow_state.get("ready_tasks", []),
            blocked_tasks=workflow_state.get("blocked_tasks", []),
            performance_metrics={
                "execution_initiation_time_ms": (datetime.utcnow() - start_time).total_seconds() * 1000,
                "execution_id": execution_result.get("execution_id")
            }
        )
        
        # Performance monitoring
        execution_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        if execution_time_ms > 500:
            logger.warning("Workflow execution exceeded performance target",
                          workflow_id=workflow_id,
                          execution_time_ms=execution_time_ms,
                          target_ms=500)
        
        logger.info("Workflow execution initiated successfully",
                   workflow_id=workflow_id,
                   execution_time_ms=execution_time_ms)
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Workflow execution failed",
                    workflow_id=workflow_id,
                    error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Workflow execution failed: {str(e)}"
        )


@router.get("/{workflow_id}/progress", response_model=WorkflowExecutionResponse)
@with_circuit_breaker(workflow_circuit_breaker) 
async def get_workflow_progress(
    workflow_id: str = Path(..., description="Workflow ID"),
    user: Dict[str, Any] = Depends(require_task_read)
) -> WorkflowExecutionResponse:
    """
    Get comprehensive workflow progress with real-time state tracking.
    
    Features:
    - Real-time progress tracking with Redis caching
    - Task-level state monitoring
    - Performance metrics and analytics
    - Dependency resolution status
    - Performance target: <100ms progress retrieval
    
    Args:
        workflow_id: Unique workflow identifier
        user: Authenticated user information
        
    Returns:
        WorkflowExecutionResponse with detailed progress information
    """
    start_time = datetime.utcnow()
    
    try:
        # Validate workflow ID
        try:
            uuid.UUID(workflow_id)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid workflow ID format"
            )
        
        # Get detailed progress through service
        progress_result = await workflow_service.get_workflow_progress_detailed(workflow_id)
        
        # Build response from progress data
        response = WorkflowExecutionResponse(
            workflow_id=workflow_id,
            name=progress_result.get("name", "Unknown Workflow"),
            description=progress_result.get("description"),
            status=progress_result.get("status", WorkflowStatus.UNKNOWN.value),
            priority=progress_result.get("priority", WorkflowPriority.MEDIUM.value),
            completion_percentage=progress_result["progress"].get("completion_percentage", 0.0),
            total_tasks=progress_result.get("total_tasks", 0),
            completed_tasks=progress_result["progress"].get("completed_tasks", 0),
            failed_tasks=progress_result["progress"].get("failed_tasks", 0),
            pending_tasks=progress_result["progress"].get("pending_tasks", 0),
            created_at=datetime.fromisoformat(progress_result.get("created_at", datetime.utcnow().isoformat())),
            started_at=datetime.fromisoformat(progress_result.get("started_at")) if progress_result.get("started_at") else None,
            completed_at=datetime.fromisoformat(progress_result.get("completed_at")) if progress_result.get("completed_at") else None,
            estimated_completion=datetime.fromisoformat(progress_result["estimated_completion"]) if progress_result.get("estimated_completion") else None,
            current_tasks=progress_result.get("current_tasks", []),
            ready_tasks=progress_result.get("ready_tasks", []),
            blocked_tasks=progress_result.get("blocked_tasks", []),
            performance_metrics={
                **progress_result.get("performance_metrics", {}),
                "progress_retrieval_time_ms": (datetime.utcnow() - start_time).total_seconds() * 1000
            }
        )
        
        # Performance monitoring
        retrieval_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        logger.info("Workflow progress retrieved",
                   workflow_id=workflow_id,
                   completion_percentage=response.completion_percentage,
                   retrieval_time_ms=retrieval_time_ms)
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Workflow progress retrieval failed",
                    workflow_id=workflow_id,
                    error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Progress retrieval failed: {str(e)}"
        )


@router.post("/{workflow_id}/pause", response_model=OperationResponse)
@audit_operation("pause_workflow", "workflow")
@with_circuit_breaker(workflow_circuit_breaker)
async def pause_workflow(
    workflow_id: str = Path(..., description="Workflow ID"),
    reason: Optional[str] = Query(None, description="Reason for pausing"),
    user: Dict[str, Any] = Depends(require_workflow_execute)
) -> OperationResponse:
    """
    Pause a running workflow with state preservation.
    
    Args:
        workflow_id: Unique workflow identifier  
        reason: Reason for pausing
        user: Authenticated user information
        
    Returns:
        OperationResponse with pause status
    """
    start_time = datetime.utcnow()
    
    try:
        # Validate workflow ID
        try:
            uuid.UUID(workflow_id)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid workflow ID format"
            )
        
        logger.info("Pausing workflow",
                   workflow_id=workflow_id,
                   reason=reason,
                   user_id=user.get("user_id"))
        
        # Pause workflow through orchestrator
        pause_result = await workflow_service.orchestrator.pause_workflow(
            workflow_id=workflow_id,
            reason=reason,
            paused_by=user.get("user_id")
        )
        
        if not pause_result.get("success", False):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=pause_result.get("message", "Workflow pause failed")
            )
        
        # Update workflow state cache
        if workflow_service.redis_service:
            workflow_state = await workflow_service.redis_service.cache_get(f"workflow:{workflow_id}")
            if workflow_state:
                workflow_state["status"] = WorkflowStatus.PAUSED.value
                workflow_state["paused_at"] = start_time.isoformat()
                workflow_state["pause_reason"] = reason
                await workflow_service.redis_service.cache_set(f"workflow:{workflow_id}", workflow_state, ttl=86400)
                
                # Publish pause event
                await workflow_service.redis_service.publish("workflow_events", {
                    "event": "workflow_paused",
                    "workflow_id": workflow_id,
                    "reason": reason,
                    "paused_by": user.get("user_id"),
                    "timestamp": start_time.isoformat()
                })
        
        response = OperationResponse(
            success=True,
            message=f"Workflow {workflow_id} paused successfully",
            operation_id=str(uuid.uuid4()),
            timestamp=start_time,
            details={
                "workflow_id": workflow_id,
                "reason": reason,
                "paused_by": user.get("user_id"),
                "running_tasks_preserved": pause_result.get("running_tasks_count", 0)
            }
        )
        
        logger.info("Workflow paused successfully",
                   workflow_id=workflow_id,
                   reason=reason)
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Workflow pause failed",
                    workflow_id=workflow_id,
                    error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Workflow pause failed: {str(e)}"
        )


@router.post("/{workflow_id}/resume", response_model=OperationResponse)
@audit_operation("resume_workflow", "workflow")
@with_circuit_breaker(workflow_circuit_breaker)
async def resume_workflow(
    workflow_id: str = Path(..., description="Workflow ID"),
    user: Dict[str, Any] = Depends(require_workflow_execute)
) -> OperationResponse:
    """
    Resume a paused workflow with state restoration.
    
    Args:
        workflow_id: Unique workflow identifier
        user: Authenticated user information
        
    Returns:
        OperationResponse with resume status
    """
    start_time = datetime.utcnow()
    
    try:
        # Validate workflow ID
        try:
            uuid.UUID(workflow_id)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid workflow ID format"
            )
        
        logger.info("Resuming workflow",
                   workflow_id=workflow_id,
                   user_id=user.get("user_id"))
        
        # Resume workflow through orchestrator
        resume_result = await workflow_service.orchestrator.resume_workflow(
            workflow_id=workflow_id,
            resumed_by=user.get("user_id")
        )
        
        if not resume_result.get("success", False):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=resume_result.get("message", "Workflow resume failed")
            )
        
        # Update workflow state cache
        if workflow_service.redis_service:
            workflow_state = await workflow_service.redis_service.cache_get(f"workflow:{workflow_id}")
            if workflow_state:
                workflow_state["status"] = WorkflowStatus.RUNNING.value
                workflow_state["resumed_at"] = start_time.isoformat()
                workflow_state.pop("pause_reason", None)
                await workflow_service.redis_service.cache_set(f"workflow:{workflow_id}", workflow_state, ttl=86400)
                
                # Publish resume event
                await workflow_service.redis_service.publish("workflow_events", {
                    "event": "workflow_resumed",
                    "workflow_id": workflow_id,
                    "resumed_by": user.get("user_id"),
                    "timestamp": start_time.isoformat()
                })
        
        response = OperationResponse(
            success=True,
            message=f"Workflow {workflow_id} resumed successfully",
            operation_id=str(uuid.uuid4()),
            timestamp=start_time,
            details={
                "workflow_id": workflow_id,
                "resumed_by": user.get("user_id"),
                "tasks_to_resume": resume_result.get("tasks_to_resume", [])
            }
        )
        
        logger.info("Workflow resumed successfully", workflow_id=workflow_id)
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Workflow resume failed",
                    workflow_id=workflow_id,
                    error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Workflow resume failed: {str(e)}"
        )


@router.post("/{workflow_id}/tasks", response_model=OperationResponse)
@audit_operation("add_task_to_workflow", "workflow")
@with_circuit_breaker(workflow_circuit_breaker)
async def add_task_to_workflow(
    workflow_id: str = Path(..., description="Workflow ID"),
    task_assignment: WorkflowTaskAssignmentRequest = ...,
    user: Dict[str, Any] = Depends(require_workflow_execute)
) -> OperationResponse:
    """
    Add a task to an existing workflow with dependency management.
    
    Args:
        workflow_id: Unique workflow identifier
        task_assignment: Task assignment parameters
        user: Authenticated user information
        
    Returns:
        OperationResponse with assignment status
    """
    start_time = datetime.utcnow()
    
    try:
        # Validate IDs
        try:
            uuid.UUID(workflow_id)
            uuid.UUID(task_assignment.task_id)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid workflow or task ID format"
            )
        
        logger.info("Adding task to workflow",
                   workflow_id=workflow_id,
                   task_id=task_assignment.task_id,
                   user_id=user.get("user_id"))
        
        # Add task through orchestrator
        add_result = await workflow_service.orchestrator.add_task_to_workflow(
            workflow_id=workflow_id,
            task_id=task_assignment.task_id,
            dependencies=task_assignment.dependencies,
            parallel_eligible=task_assignment.parallel_eligible,
            critical_path=task_assignment.critical_path,
            added_by=user.get("user_id")
        )
        
        if not add_result.get("success", False):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=add_result.get("message", "Failed to add task to workflow")
            )
        
        # Update workflow state cache
        if workflow_service.redis_service:
            workflow_state = await workflow_service.redis_service.cache_get(f"workflow:{workflow_id}")
            if workflow_state:
                workflow_state["total_tasks"] = workflow_state.get("total_tasks", 0) + 1
                workflow_state["updated_at"] = start_time.isoformat()
                await workflow_service.redis_service.cache_set(f"workflow:{workflow_id}", workflow_state, ttl=86400)
                
                # Publish task added event
                await workflow_service.redis_service.publish("workflow_events", {
                    "event": "task_added_to_workflow",
                    "workflow_id": workflow_id,
                    "task_id": task_assignment.task_id,
                    "dependencies_count": len(task_assignment.dependencies or []),
                    "added_by": user.get("user_id"),
                    "timestamp": start_time.isoformat()
                })
        
        response = OperationResponse(
            success=True,
            message=f"Task {task_assignment.task_id} added to workflow {workflow_id}",
            operation_id=str(uuid.uuid4()),
            timestamp=start_time,
            details={
                "workflow_id": workflow_id,
                "task_id": task_assignment.task_id,
                "dependencies": task_assignment.dependencies,
                "parallel_eligible": task_assignment.parallel_eligible,
                "critical_path": task_assignment.critical_path
            }
        )
        
        logger.info("Task added to workflow successfully",
                   workflow_id=workflow_id,
                   task_id=task_assignment.task_id)
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to add task to workflow",
                    workflow_id=workflow_id,
                    task_id=task_assignment.task_id,
                    error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to add task to workflow: {str(e)}"
        )


# ===============================================================================
# BACKGROUND MONITORING FUNCTIONS
# ===============================================================================

async def monitor_workflow_execution(workflow_id: str, user_id: str):
    """Background task to monitor workflow execution progress."""
    try:
        logger.info("Starting workflow execution monitoring",
                   workflow_id=workflow_id,
                   user_id=user_id)
        
        # Monitor workflow progress periodically
        monitoring_interval = 30  # seconds
        max_monitoring_duration = 3600  # 1 hour
        start_time = datetime.utcnow()
        
        while (datetime.utcnow() - start_time).total_seconds() < max_monitoring_duration:
            try:
                # Check workflow progress
                progress_data = await workflow_service.get_workflow_progress_detailed(workflow_id)
                status = progress_data.get("status")
                
                # Stop monitoring if workflow completed or failed
                if status in [WorkflowStatus.COMPLETED.value, WorkflowStatus.FAILED.value, WorkflowStatus.CANCELLED.value]:
                    logger.info("Workflow monitoring completed",
                               workflow_id=workflow_id,
                               final_status=status)
                    break
                
                # Log progress update
                completion_percentage = progress_data["progress"].get("completion_percentage", 0.0)
                logger.info("Workflow progress update",
                           workflow_id=workflow_id,
                           status=status,
                           completion_percentage=completion_percentage)
                
                # Wait before next check
                await asyncio.sleep(monitoring_interval)
                
            except Exception as e:
                logger.warning("Workflow monitoring check failed",
                              workflow_id=workflow_id,
                              error=str(e))
                await asyncio.sleep(monitoring_interval)
        
        logger.info("Workflow monitoring ended",
                   workflow_id=workflow_id,
                   duration_seconds=(datetime.utcnow() - start_time).total_seconds())
        
    except Exception as e:
        logger.error("Workflow monitoring failed",
                    workflow_id=workflow_id,
                    error=str(e))