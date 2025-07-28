"""
Agent Orchestrator Core API endpoints for LeanVibe Agent Hive 2.0

Provides FastAPI async engine endpoints for comprehensive agent orchestration,
task scheduling, health monitoring, and system management with proper error handling
and performance optimization according to PRD specifications.
"""

import asyncio
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, Field

import structlog

from ...core.database import get_session_dependency
from ...core.agent_registry import AgentRegistry, AgentRegistrationResult, LifecycleState
from ...core.task_queue import TaskQueue, QueuedTask, QueueStatus
from ...core.task_scheduler import TaskScheduler, SchedulingStrategy, SchedulingDecision
from ...core.health_monitor import HealthMonitor, HealthStatus, CheckType
from ...models.agent import Agent, AgentStatus, AgentType
from ...models.task import Task, TaskStatus, TaskPriority, TaskType

logger = structlog.get_logger()
router = APIRouter()

# Global service instances (would be dependency-injected in production)
_agent_registry: Optional[AgentRegistry] = None
_task_queue: Optional[TaskQueue] = None
_task_scheduler: Optional[TaskScheduler] = None
_health_monitor: Optional[HealthMonitor] = None


# Pydantic models for API requests/responses
class AgentRegistrationRequest(BaseModel):
    """Request model for agent registration."""
    name: str = Field(..., description="Agent name")
    agent_type: AgentType = Field(default=AgentType.CLAUDE, description="Agent type")
    role: Optional[str] = Field(None, description="Agent role/specialization")
    capabilities: Optional[List[Dict[str, Any]]] = Field(default_factory=list, description="Agent capabilities")
    system_prompt: Optional[str] = Field(None, description="Agent system prompt")
    config: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Agent configuration")
    tmux_session: Optional[str] = Field(None, description="Associated tmux session")


class AgentRegistrationResponse(BaseModel):
    """Response model for agent registration."""
    success: bool
    agent_id: Optional[str]
    error_message: Optional[str]
    registration_time: Optional[datetime]
    capabilities_assigned: List[str]
    health_score: float


class TaskSubmissionRequest(BaseModel):
    """Request model for task submission."""
    title: str = Field(..., description="Task title")
    description: Optional[str] = Field(None, description="Task description")
    task_type: Optional[TaskType] = Field(None, description="Task type")
    priority: TaskPriority = Field(default=TaskPriority.MEDIUM, description="Task priority")
    required_capabilities: Optional[List[str]] = Field(default_factory=list, description="Required capabilities")
    estimated_effort: Optional[int] = Field(None, description="Estimated effort in minutes")
    timeout_seconds: Optional[int] = Field(None, description="Task timeout in seconds")
    context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Task context")


class TaskAssignmentRequest(BaseModel):
    """Request model for task assignment."""
    task_id: str = Field(..., description="Task ID to assign")
    strategy: Optional[SchedulingStrategy] = Field(None, description="Scheduling strategy")
    preferred_agent_id: Optional[str] = Field(None, description="Preferred agent ID")
    timeout_seconds: Optional[float] = Field(None, description="Assignment timeout")


class TaskCompletionRequest(BaseModel):
    """Request model for task completion."""
    task_id: str = Field(..., description="Task ID to complete")
    agent_id: str = Field(..., description="Agent ID that completed the task")
    success: bool = Field(default=True, description="Whether task completed successfully")
    result: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Task result")
    execution_time_seconds: Optional[float] = Field(None, description="Execution time in seconds")


async def get_agent_registry() -> AgentRegistry:
    """Get or create agent registry instance."""
    global _agent_registry
    if _agent_registry is None:
        _agent_registry = AgentRegistry()
        await _agent_registry.start()
    return _agent_registry


async def get_task_queue() -> TaskQueue:
    """Get or create task queue instance."""
    global _task_queue
    if _task_queue is None:
        _task_queue = TaskQueue()
        await _task_queue.start()
    return _task_queue


async def get_task_scheduler() -> TaskScheduler:
    """Get or create task scheduler instance."""
    global _task_scheduler
    if _task_scheduler is None:
        registry = await get_agent_registry()
        queue = await get_task_queue()
        _task_scheduler = TaskScheduler(registry, queue)
        await _task_scheduler.start()
    return _task_scheduler


async def get_health_monitor() -> HealthMonitor:
    """Get or create health monitor instance."""
    global _health_monitor
    if _health_monitor is None:
        _health_monitor = HealthMonitor()
        await _health_monitor.start()
    return _health_monitor


# === AGENT MANAGEMENT ENDPOINTS ===

@router.post("/agents/register", response_model=AgentRegistrationResponse, status_code=201)
async def register_agent(
    request: AgentRegistrationRequest,
    background_tasks: BackgroundTasks,
    registry: AgentRegistry = Depends(get_agent_registry)
) -> AgentRegistrationResponse:
    """
    Register a new AI agent with the orchestrator.
    
    This endpoint provides enhanced agent registration with lifecycle tracking,
    capability assignment, and performance monitoring setup.
    
    Performance target: <10 seconds registration time
    """
    registration_start = datetime.utcnow()
    
    try:
        # Register agent with enhanced capabilities
        result = await registry.register_agent(
            name=request.name,
            agent_type=request.agent_type,
            role=request.role,
            capabilities=request.capabilities,
            system_prompt=request.system_prompt,
            config=request.config,
            tmux_session=request.tmux_session
        )
        
        # Schedule background health monitoring setup
        if result.success:
            background_tasks.add_task(
                _setup_agent_monitoring,
                result.agent_id
            )
        
        # Calculate registration time
        registration_time = (datetime.utcnow() - registration_start).total_seconds()
        
        logger.info(
            "Agent registration completed",
            agent_id=str(result.agent_id) if result.agent_id else None,
            success=result.success,
            registration_time_seconds=registration_time
        )
        
        return AgentRegistrationResponse(
            success=result.success,
            agent_id=str(result.agent_id) if result.agent_id else None,
            error_message=result.error_message,
            registration_time=result.registration_time,
            capabilities_assigned=result.capabilities_assigned,
            health_score=result.health_score
        )
        
    except Exception as e:
        logger.error("Agent registration failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Agent registration failed: {str(e)}"
        )


@router.get("/agents/{agent_id}")
async def get_agent_details(
    agent_id: str,
    registry: AgentRegistry = Depends(get_agent_registry)
) -> Dict[str, Any]:
    """Get comprehensive agent details including health and performance metrics."""
    try:
        agent_uuid = uuid.UUID(agent_id)
        agent = await registry.get_agent(agent_uuid)
        
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        # Get health monitor for additional metrics
        health_monitor = await get_health_monitor()
        health_overview = await health_monitor.get_system_health_overview()
        
        return {
            "agent": agent.to_dict(),
            "system_health": {
                "total_agents": health_overview.total_agents,
                "healthy_agents": health_overview.healthy_agents,
                "average_health_score": health_overview.average_health_score
            },
            "retrieved_at": datetime.utcnow().isoformat()
        }
        
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid agent ID format")
    except Exception as e:
        logger.error("Failed to get agent details", agent_id=agent_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve agent details")


@router.get("/agents")
async def list_agents(
    status: Optional[AgentStatus] = Query(None, description="Filter by agent status"),
    lifecycle_state: Optional[str] = Query(None, description="Filter by lifecycle state"),
    role: Optional[str] = Query(None, description="Filter by agent role"),
    limit: int = Query(50, ge=1, le=100, description="Number of agents to return"),
    offset: int = Query(0, ge=0, description="Number of agents to skip"),
    registry: AgentRegistry = Depends(get_agent_registry)
) -> Dict[str, Any]:
    """List agents with filtering and pagination."""
    try:
        # Convert lifecycle state string to enum if provided
        lifecycle_enum = None
        if lifecycle_state:
            try:
                lifecycle_enum = LifecycleState(lifecycle_state)
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid lifecycle state: {lifecycle_state}"
                )
        
        agents = await registry.list_agents(
            status=status,
            lifecycle_state=lifecycle_enum,
            role=role,
            limit=limit,
            offset=offset
        )
        
        # Get registry statistics
        stats = await registry.get_agent_statistics()
        
        return {
            "agents": [agent.to_dict() for agent in agents],
            "pagination": {
                "offset": offset,
                "limit": limit,
                "total_returned": len(agents)
            },
            "statistics": stats,
            "retrieved_at": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to list agents", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to list agents")


@router.delete("/agents/{agent_id}")
async def deregister_agent(
    agent_id: str,
    graceful: bool = Query(True, description="Whether to perform graceful shutdown"),
    registry: AgentRegistry = Depends(get_agent_registry)
) -> Dict[str, Any]:
    """Deregister an agent with proper cleanup."""
    try:
        agent_uuid = uuid.UUID(agent_id)
        success = await registry.deregister_agent(agent_uuid, graceful)
        
        if not success:
            raise HTTPException(status_code=404, detail="Agent not found or deregistration failed")
        
        return {
            "success": True,
            "agent_id": agent_id,
            "graceful": graceful,
            "deregistered_at": datetime.utcnow().isoformat()
        }
        
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid agent ID format")
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to deregister agent", agent_id=agent_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to deregister agent")


# === TASK MANAGEMENT ENDPOINTS ===

@router.post("/tasks/submit", status_code=201)
async def submit_task(
    request: TaskSubmissionRequest,
    auto_assign: bool = Query(True, description="Automatically assign task to agent"),
    db: AsyncSession = Depends(get_session_dependency),
    task_queue: TaskQueue = Depends(get_task_queue)
) -> Dict[str, Any]:
    """
    Submit a new task to the orchestrator.
    
    This endpoint creates a task and optionally queues it for automatic assignment.
    Performance target: <500ms API response time
    """
    submission_start = datetime.utcnow()
    
    try:
        # Create task in database
        task = Task(
            title=request.title,
            description=request.description,
            task_type=request.task_type,
            priority=request.priority,
            required_capabilities=request.required_capabilities or [],
            estimated_effort=request.estimated_effort,
            timeout_seconds=request.timeout_seconds,
            context=request.context or {},
            orchestrator_metadata={
                "submitted_at": submission_start.isoformat(),
                "auto_assign": auto_assign,
                "orchestrator_version": "3.1.0"
            }
        )
        
        db.add(task)
        await db.commit()
        await db.refresh(task)
        
        # Enqueue task for assignment if requested
        task_queued = False
        if auto_assign:
            task_queued = await task_queue.enqueue_task(
                task_id=task.id,
                priority=request.priority,
                required_capabilities=request.required_capabilities,
                estimated_effort=request.estimated_effort,
                timeout_seconds=request.timeout_seconds,
                metadata={"auto_assign": True}
            )
        
        # Calculate response time
        response_time = (datetime.utcnow() - submission_start).total_seconds() * 1000
        
        logger.info(
            "Task submitted successfully",
            task_id=str(task.id),
            auto_assign=auto_assign,
            task_queued=task_queued,
            response_time_ms=response_time
        )
        
        return {
            "success": True,
            "task_id": str(task.id),
            "task_queued": task_queued,
            "estimated_assignment_time_seconds": 30 if task_queued else None,
            "response_time_ms": response_time,
            "submitted_at": submission_start.isoformat()
        }
        
    except Exception as e:
        logger.error("Task submission failed", error=str(e))
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"Task submission failed: {str(e)}")


@router.post("/tasks/assign")
async def assign_task(
    request: TaskAssignmentRequest,
    scheduler: TaskScheduler = Depends(get_task_scheduler)
) -> Dict[str, Any]:
    """
    Assign a task to an appropriate agent using intelligent scheduling.
    
    Performance target: <2 seconds assignment time
    """
    assignment_start = datetime.utcnow()
    
    try:
        task_uuid = uuid.UUID(request.task_id)
        preferred_agent_uuid = None
        
        if request.preferred_agent_id:
            preferred_agent_uuid = uuid.UUID(request.preferred_agent_id)
        
        # Perform intelligent task assignment
        decision = await scheduler.assign_task(
            task_id=task_uuid,
            strategy=request.strategy,
            preferred_agent_id=preferred_agent_uuid,
            timeout_seconds=request.timeout_seconds
        )
        
        response_time = (datetime.utcnow() - assignment_start).total_seconds() * 1000
        
        if decision.success:
            logger.info(
                "Task assigned successfully",
                task_id=request.task_id,
                agent_id=str(decision.assigned_agent_id),
                strategy=decision.scheduling_strategy.value,
                confidence=decision.assignment_confidence,
                response_time_ms=response_time
            )
        else:
            logger.warning(
                "Task assignment failed",
                task_id=request.task_id,
                error=decision.error_message,
                response_time_ms=response_time
            )
        
        return {
            "success": decision.success,
            "task_id": request.task_id,
            "assigned_agent_id": str(decision.assigned_agent_id) if decision.assigned_agent_id else None,
            "assignment_confidence": decision.assignment_confidence,
            "scheduling_strategy": decision.scheduling_strategy.value,
            "decision_time_ms": decision.decision_time_ms,
            "reasoning": decision.reasoning,
            "error_message": decision.error_message,
            "assigned_at": datetime.utcnow().isoformat() if decision.success else None
        }
        
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid task or agent ID format")
    except Exception as e:
        logger.error("Task assignment failed", task_id=request.task_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Task assignment failed: {str(e)}")


@router.post("/tasks/complete")
async def complete_task(
    request: TaskCompletionRequest,
    db: AsyncSession = Depends(get_session_dependency)
) -> Dict[str, Any]:
    """Complete a task and update metrics."""
    try:
        task_uuid = uuid.UUID(request.task_id)
        agent_uuid = uuid.UUID(request.agent_id)
        
        # Get task from database
        result = await db.execute(
            db.select(Task).where(Task.id == task_uuid)
        )
        task = result.scalar_one_or_none()
        
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        
        if request.success:
            task.complete_successfully(request.result or {})
        else:
            task.fail_with_error(
                request.result.get("error_message", "Task failed") if request.result else "Task failed",
                can_retry=True
            )
        
        # Update execution metadata
        task.execution_metadata = {
            **(task.execution_metadata or {}),
            "execution_time_seconds": request.execution_time_seconds,
            "completed_by_agent": request.agent_id,
            "completion_timestamp": datetime.utcnow().isoformat()
        }
        
        await db.commit()
        
        logger.info(
            "Task completed",
            task_id=request.task_id,
            agent_id=request.agent_id,
            success=request.success,
            execution_time=request.execution_time_seconds
        )
        
        return {
            "success": True,
            "task_id": request.task_id,
            "agent_id": request.agent_id,
            "task_success": request.success,
            "completed_at": datetime.utcnow().isoformat()
        }
        
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid task or agent ID format")
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Task completion failed", task_id=request.task_id, error=str(e))
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"Task completion failed: {str(e)}")


@router.get("/tasks/{task_id}")
async def get_task_status(
    task_id: str,
    db: AsyncSession = Depends(get_session_dependency)
) -> Dict[str, Any]:
    """Get comprehensive task status and execution details."""
    try:
        task_uuid = uuid.UUID(task_id)
        
        result = await db.execute(
            db.select(Task).where(Task.id == task_uuid)
        )
        task = result.scalar_one_or_none()
        
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        
        return {
            "task": task.to_dict(),
            "retrieved_at": datetime.utcnow().isoformat()
        }
        
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid task ID format")
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get task status", task_id=task_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve task status")


# === HEALTH MONITORING ENDPOINTS ===

@router.get("/health/system")
async def get_system_health(
    health_monitor: HealthMonitor = Depends(get_health_monitor)
) -> Dict[str, Any]:
    """Get comprehensive system health overview."""
    try:
        overview = await health_monitor.get_system_health_overview()
        
        return {
            "system_health": {
                "total_agents": overview.total_agents,
                "healthy_agents": overview.healthy_agents,
                "warning_agents": overview.warning_agents,
                "critical_agents": overview.critical_agents,
                "failed_agents": overview.failed_agents,
                "average_health_score": overview.average_health_score,
                "system_uptime_hours": overview.system_uptime_hours,
                "recent_alerts": overview.recent_alerts,
                "performance_summary": overview.performance_summary
            },
            "checked_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error("Failed to get system health", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve system health")


@router.post("/health/agents/{agent_id}/check")
async def check_agent_health(
    agent_id: str,
    check_type: str = Query("comprehensive", description="Type of health check"),
    health_monitor: HealthMonitor = Depends(get_health_monitor)
) -> Dict[str, Any]:
    """Perform a health check on a specific agent."""
    try:
        agent_uuid = uuid.UUID(agent_id)
        
        # Parse check type
        try:
            check_enum = CheckType(check_type.lower())
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid check type: {check_type}"
            )
        
        result = await health_monitor.check_agent_health(agent_uuid, check_enum)
        
        return {
            "agent_id": agent_id,
            "check_type": result.check_type.value,
            "status": result.status.value,
            "score": result.score,
            "response_time_ms": result.response_time_ms,
            "data": result.data,
            "error_message": result.error_message,
            "checked_at": result.checked_at.isoformat()
        }
        
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid agent ID format")
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Health check failed", agent_id=agent_id, error=str(e))
        raise HTTPException(status_code=500, detail="Health check failed")


# === QUEUE MANAGEMENT ENDPOINTS ===

@router.get("/queue/stats")
async def get_queue_statistics(
    queue_name: Optional[str] = Query(None, description="Specific queue name"),
    task_queue: TaskQueue = Depends(get_task_queue)
) -> Dict[str, Any]:
    """Get comprehensive queue statistics and performance metrics."""
    try:
        stats = await task_queue.get_queue_stats(queue_name)
        
        return {
            "queue_statistics": stats,
            "retrieved_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error("Failed to get queue statistics", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve queue statistics")


# === SCHEDULER STATUS ENDPOINTS ===

@router.get("/scheduler/stats")
async def get_scheduler_statistics(
    scheduler: TaskScheduler = Depends(get_task_scheduler)
) -> Dict[str, Any]:
    """Get comprehensive scheduling statistics and performance metrics."""
    try:
        stats = await scheduler.get_scheduling_statistics()
        
        return {
            "scheduling_statistics": stats,
            "retrieved_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error("Failed to get scheduler statistics", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve scheduler statistics")


# === WORKFLOW DEMONSTRATION ENDPOINT ===

@router.post("/demo/single-task-workflow")
async def demonstrate_single_task_workflow(
    task_title: str = Query("Demo Task", description="Title for the demo task"),
    auto_complete: bool = Query(True, description="Auto-complete the task for demo"),
    db: AsyncSession = Depends(get_session_dependency),
    registry: AgentRegistry = Depends(get_agent_registry),
    task_queue: TaskQueue = Depends(get_task_queue),
    scheduler: TaskScheduler = Depends(get_task_scheduler)
) -> Dict[str, Any]:
    """
    Demonstrate complete single-task workflow: submission → assignment → completion.
    
    This endpoint showcases the full orchestrator capability in a single operation.
    """
    workflow_start = datetime.utcnow()
    workflow_log = []
    
    try:
        # Step 1: Submit task
        task_submission = TaskSubmissionRequest(
            title=task_title,
            description="Demonstration task for orchestrator core validation",
            task_type=TaskType.TESTING,
            priority=TaskPriority.HIGH,
            required_capabilities=["testing", "demonstration"],
            estimated_effort=5,
            context={"demo": True, "workflow_start": workflow_start.isoformat()}
        )
        
        # Create task
        task = Task(
            title=task_submission.title,
            description=task_submission.description,
            task_type=task_submission.task_type,
            priority=task_submission.priority,
            required_capabilities=task_submission.required_capabilities,
            estimated_effort=task_submission.estimated_effort,
            context=task_submission.context,
            orchestrator_metadata={
                "demo_workflow": True,
                "submitted_at": datetime.utcnow().isoformat()
            }
        )
        
        db.add(task)
        await db.commit()
        await db.refresh(task)
        
        workflow_log.append({
            "step": "task_submission",
            "timestamp": datetime.utcnow().isoformat(),
            "task_id": str(task.id),
            "status": "completed"
        })
        
        # Step 2: Assign task
        decision = await scheduler.assign_task(
            task_id=task.id,
            strategy=SchedulingStrategy.HYBRID,
            timeout_seconds=2.0
        )
        
        workflow_log.append({
            "step": "task_assignment",
            "timestamp": datetime.utcnow().isoformat(),
            "success": decision.success,
            "assigned_agent_id": str(decision.assigned_agent_id) if decision.assigned_agent_id else None,
            "confidence": decision.assignment_confidence,
            "strategy": decision.scheduling_strategy.value
        })
        
        # Step 3: Complete task (if auto_complete and assignment successful)
        task_completed = False
        if auto_complete and decision.success:
            # Simulate task execution
            await asyncio.sleep(0.1)  # Brief simulation
            
            task.complete_successfully({
                "demo_result": "Task completed successfully",
                "execution_details": {
                    "simulated": True,
                    "execution_time": 0.1,
                    "agent_id": str(decision.assigned_agent_id)
                }
            })
            
            await db.commit()
            task_completed = True
            
            workflow_log.append({
                "step": "task_completion",
                "timestamp": datetime.utcnow().isoformat(),
                "success": True,
                "simulated": True
            })
        
        # Calculate total workflow time
        workflow_duration = (datetime.utcnow() - workflow_start).total_seconds() * 1000
        
        logger.info(
            "Single-task workflow demonstration completed",
            task_id=str(task.id),
            assignment_success=decision.success,
            task_completed=task_completed,
            workflow_duration_ms=workflow_duration
        )
        
        return {
            "success": True,
            "workflow_type": "single_task_demonstration",
            "task_id": str(task.id),
            "assignment_success": decision.success,
            "task_completed": task_completed,
            "workflow_duration_ms": workflow_duration,
            "workflow_log": workflow_log,
            "performance_metrics": {
                "task_submission_time_ms": workflow_log[0].get("duration_ms", 0),
                "assignment_time_ms": decision.decision_time_ms,
                "total_workflow_time_ms": workflow_duration
            },
            "completed_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error("Single-task workflow demonstration failed", error=str(e))
        await db.rollback()
        
        workflow_log.append({
            "step": "workflow_error",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e)
        })
        
        return {
            "success": False,
            "error_message": str(e),
            "workflow_log": workflow_log,
            "failed_at": datetime.utcnow().isoformat()
        }


# === BACKGROUND TASKS ===

async def _setup_agent_monitoring(agent_id: uuid.UUID) -> None:
    """Setup background monitoring for a newly registered agent."""
    try:
        health_monitor = await get_health_monitor()
        
        # Perform initial health check
        await health_monitor.check_agent_health(agent_id, CheckType.COMPREHENSIVE)
        
        logger.info("Agent monitoring setup completed", agent_id=str(agent_id))
        
    except Exception as e:
        logger.error("Failed to setup agent monitoring", agent_id=str(agent_id), error=str(e))