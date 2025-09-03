"""
Epic B: Task Management API Endpoints

Implements basic task creation and assignment functionality
for the Mobile PWA. Supports task distribution and monitoring
as part of the core orchestration system.
"""

import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel, Field
import structlog

from app.core.simple_orchestrator import SimpleOrchestrator, create_simple_orchestrator, TaskAssignment
from app.models.task import TaskStatus, TaskPriority

logger = structlog.get_logger()
router = APIRouter(prefix="/tasks", tags=["tasks"])

# Get orchestrator instance
async def get_orchestrator() -> SimpleOrchestrator:
    """Get orchestrator instance."""
    from .agents import get_orchestrator
    return await get_orchestrator()


# Pydantic models for tasks
class TaskCreateRequest(BaseModel):
    """Request model for task creation."""
    description: str = Field(..., description="Task description", min_length=1, max_length=1000)
    task_type: str = Field(default="general", description="Type of task")
    priority: str = Field(default="medium", description="Task priority (low, medium, high)")
    agent_id: Optional[str] = Field(None, description="Specific agent to assign to")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional task context")


class TaskResponse(BaseModel):
    """Response model for task data."""
    id: str
    description: str
    task_type: str
    priority: str
    status: str
    agent_id: Optional[str] = None
    created_at: str
    assigned_at: Optional[str] = None
    context: Optional[Dict[str, Any]] = None


class TaskListResponse(BaseModel):
    """Response model for task list."""
    tasks: List[TaskResponse]
    total: int
    pending: int
    in_progress: int
    completed: int


class TaskStatusUpdate(BaseModel):
    """Request model for task status updates."""
    status: str = Field(..., description="New task status")
    result: Optional[Dict[str, Any]] = Field(None, description="Task result data")


@router.post("/", response_model=TaskResponse, status_code=201)
async def create_task(
    request: TaskCreateRequest,
    background_tasks: BackgroundTasks
) -> TaskResponse:
    """
    Create a new task and optionally assign to agent.
    
    Epic B Phase B.2: Task creation and distribution functionality.
    """
    try:
        orchestrator = await get_orchestrator()
        
        # Validate priority
        try:
            # Map string names to TaskPriority enum
            priority_mapping = {
                "LOW": TaskPriority.LOW,
                "MEDIUM": TaskPriority.MEDIUM, 
                "HIGH": TaskPriority.HIGH,
                "CRITICAL": TaskPriority.CRITICAL
            }
            priority = priority_mapping.get(request.priority.upper())
            if priority is None:
                raise ValueError("Invalid priority name")
        except (ValueError, AttributeError):
            valid_priorities = ["low", "medium", "high", "critical"]
            raise HTTPException(
                status_code=400,
                detail=f"Invalid priority. Valid priorities: {valid_priorities}"
            )
        
        # Generate task ID
        task_id = str(uuid.uuid4())
        
        # Create task assignment
        assignment = TaskAssignment(
            task_id=task_id,
            agent_id=request.agent_id or "unassigned"
        )
        
        # Store task assignment
        orchestrator._task_assignments[task_id] = assignment
        
        # If agent specified, delegate to agent
        if request.agent_id:
            agent = orchestrator._agents.get(request.agent_id)
            if not agent:
                raise HTTPException(status_code=404, detail="Agent not found")
            
            # Update agent with task
            agent.current_task_id = task_id
            assignment.agent_id = request.agent_id
            assignment.assigned_at = datetime.utcnow()
            
            # Persist task to database if available
            await orchestrator._persist_task(
                task_id=task_id,
                description=request.description,
                task_type=request.task_type,
                priority=priority,
                agent_id=request.agent_id
            )
        
        logger.info("Task created via API", 
                   task_id=task_id, 
                   agent_id=request.agent_id,
                   priority=request.priority)
        
        return TaskResponse(
            id=task_id,
            description=request.description,
            task_type=request.task_type,
            priority=priority.value,
            status=assignment.status.value,
            agent_id=request.agent_id,
            created_at=assignment.assigned_at.isoformat(),
            assigned_at=assignment.assigned_at.isoformat() if assignment.assigned_at else None,
            context=request.context
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to create task via API", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to create task")


@router.get("/", response_model=TaskListResponse)
async def list_tasks(
    status: Optional[str] = Query(None, description="Filter by status"),
    agent_id: Optional[str] = Query(None, description="Filter by agent"),
    limit: int = Query(50, ge=1, le=100, description="Max tasks to return"),
    offset: int = Query(0, ge=0, description="Tasks to skip")
) -> TaskListResponse:
    """
    List all tasks with filtering.
    
    Epic B Phase B.2: Task listing for dashboard monitoring.
    """
    try:
        orchestrator = await get_orchestrator()
        assignments = list(orchestrator._task_assignments.values())
        
        # Apply filters
        if status:
            try:
                status_enum = TaskStatus(status.upper())
                assignments = [a for a in assignments if a.status == status_enum]
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid status filter")
        
        if agent_id:
            assignments = [a for a in assignments if a.agent_id == agent_id]
        
        # Apply pagination
        total = len(assignments)
        assignments = assignments[offset:offset + limit]
        
        # Convert to response format
        task_responses = []
        for assignment in assignments:
            task_responses.append(TaskResponse(
                id=assignment.task_id,
                description=f"Task {assignment.task_id[:8]}",  # Simplified
                task_type="general",
                priority="medium",
                status=assignment.status.value,
                agent_id=assignment.agent_id if assignment.agent_id != "unassigned" else None,
                created_at=assignment.assigned_at.isoformat(),
                assigned_at=assignment.assigned_at.isoformat() if assignment.assigned_at else None
            ))
        
        # Calculate stats
        all_assignments = list(orchestrator._task_assignments.values())
        pending_count = len([a for a in all_assignments if a.status == TaskStatus.PENDING])
        in_progress_count = len([a for a in all_assignments if a.status == TaskStatus.IN_PROGRESS])
        completed_count = len([a for a in all_assignments if a.status == TaskStatus.COMPLETED])
        
        return TaskListResponse(
            tasks=task_responses,
            total=total,
            pending=pending_count,
            in_progress=in_progress_count,
            completed=completed_count
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to list tasks", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to list tasks")


@router.get("/{task_id}", response_model=TaskResponse)
async def get_task(task_id: str) -> TaskResponse:
    """
    Get specific task details.
    
    Epic B Phase B.2: Task monitoring and status tracking.
    """
    try:
        orchestrator = await get_orchestrator()
        assignment = orchestrator._task_assignments.get(task_id)
        
        if not assignment:
            raise HTTPException(status_code=404, detail="Task not found")
        
        return TaskResponse(
            id=assignment.task_id,
            description=f"Task {assignment.task_id[:8]}",
            task_type="general",
            priority="medium",
            status=assignment.status.value,
            agent_id=assignment.agent_id if assignment.agent_id != "unassigned" else None,
            created_at=assignment.assigned_at.isoformat(),
            assigned_at=assignment.assigned_at.isoformat() if assignment.assigned_at else None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get task", task_id=task_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to get task")


@router.put("/{task_id}/status", response_model=TaskResponse)
async def update_task_status(
    task_id: str,
    request: TaskStatusUpdate
) -> TaskResponse:
    """
    Update task status.
    
    Epic B Phase B.2: Task progress tracking and status updates.
    """
    try:
        orchestrator = await get_orchestrator()
        assignment = orchestrator._task_assignments.get(task_id)
        
        if not assignment:
            raise HTTPException(status_code=404, detail="Task not found")
        
        # Validate status
        try:
            new_status = TaskStatus(request.status.upper())
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid status. Valid statuses: {[s.value for s in TaskStatus]}"
            )
        
        # Update task status
        old_status = assignment.status
        assignment.status = new_status
        
        # If task completed, clear from agent
        if new_status == TaskStatus.COMPLETED and assignment.agent_id != "unassigned":
            agent = orchestrator._agents.get(assignment.agent_id)
            if agent and agent.current_task_id == task_id:
                agent.current_task_id = None
                agent.last_activity = datetime.utcnow()
        
        logger.info("Task status updated", 
                   task_id=task_id, 
                   old_status=old_status.value, 
                   new_status=new_status.value)
        
        return TaskResponse(
            id=assignment.task_id,
            description=f"Task {assignment.task_id[:8]}",
            task_type="general",
            priority="medium",
            status=assignment.status.value,
            agent_id=assignment.agent_id if assignment.agent_id != "unassigned" else None,
            created_at=assignment.assigned_at.isoformat(),
            assigned_at=assignment.assigned_at.isoformat() if assignment.assigned_at else None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to update task status", task_id=task_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to update task status")


@router.delete("/{task_id}", status_code=204)
async def delete_task(task_id: str):
    """
    Cancel and delete a task.
    
    Epic B Phase B.2: Task cancellation functionality.
    """
    try:
        orchestrator = await get_orchestrator()
        assignment = orchestrator._task_assignments.get(task_id)
        
        if not assignment:
            raise HTTPException(status_code=404, detail="Task not found")
        
        # Remove task from agent if assigned
        if assignment.agent_id != "unassigned":
            agent = orchestrator._agents.get(assignment.agent_id)
            if agent and agent.current_task_id == task_id:
                agent.current_task_id = None
                agent.last_activity = datetime.utcnow()
        
        # Remove from task assignments
        del orchestrator._task_assignments[task_id]
        
        logger.info("Task cancelled via API", task_id=task_id)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to delete task", task_id=task_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to delete task")


@router.post("/{task_id}/assign/{agent_id}", response_model=TaskResponse)
async def assign_task_to_agent(task_id: str, agent_id: str) -> TaskResponse:
    """
    Assign a task to a specific agent.
    
    Epic B Phase B.2: Task delegation and assignment functionality.
    """
    try:
        orchestrator = await get_orchestrator()
        
        # Check task exists
        assignment = orchestrator._task_assignments.get(task_id)
        if not assignment:
            raise HTTPException(status_code=404, detail="Task not found")
        
        # Check agent exists
        agent = orchestrator._agents.get(agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        # Check if agent is available
        if agent.current_task_id:
            raise HTTPException(status_code=409, detail="Agent is currently busy with another task")
        
        # Remove from current agent if assigned
        if assignment.agent_id != "unassigned":
            old_agent = orchestrator._agents.get(assignment.agent_id)
            if old_agent and old_agent.current_task_id == task_id:
                old_agent.current_task_id = None
        
        # Assign to new agent
        assignment.agent_id = agent_id
        assignment.assigned_at = datetime.utcnow()
        assignment.status = TaskStatus.IN_PROGRESS
        
        agent.current_task_id = task_id
        agent.last_activity = datetime.utcnow()
        
        logger.info("Task assigned to agent", task_id=task_id, agent_id=agent_id)
        
        return TaskResponse(
            id=assignment.task_id,
            description=f"Task {assignment.task_id[:8]}",
            task_type="general", 
            priority="medium",
            status=assignment.status.value,
            agent_id=assignment.agent_id,
            created_at=assignment.assigned_at.isoformat(),
            assigned_at=assignment.assigned_at.isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to assign task to agent", task_id=task_id, agent_id=agent_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to assign task to agent")