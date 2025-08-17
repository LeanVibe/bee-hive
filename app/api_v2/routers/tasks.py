"""
Tasks API - Consolidated task management endpoints

Consolidates dashboard_task_management.py, v1/tasks.py,
v1/consumer_groups.py, v1/dlq.py, and v1/dlq_management.py
into a unified RESTful resource for task distribution and monitoring.

Performance target: <100ms P95 response time
"""

import uuid
from typing import List, Optional, Dict, Any
from datetime import datetime

import structlog
from fastapi import APIRouter, Request, HTTPException, Query, BackgroundTasks
from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete, and_, or_
from sqlalchemy.orm import selectinload

from ...core.database import get_session_dependency
from ...core.task_execution_engine import TaskExecutionEngine
from ...models.task import Task, TaskStatus, TaskType, TaskPriority
from ...models.agent import Agent, AgentStatus
from ...schemas.task import (
    TaskCreateRequest,
    TaskUpdateRequest,
    TaskResponse,
    TaskListResponse,
    TaskAssignmentRequest,
    TaskStatsResponse
)
from ..middleware import (
    get_current_user_from_request
)

logger = structlog.get_logger()
router = APIRouter()

# Task execution engine dependency
async def get_task_engine() -> TaskExecutionEngine:
    """Get task execution engine instance."""
    return TaskExecutionEngine()

@router.post("/", response_model=TaskResponse, status_code=201)
async def create_task(
    request: Request,
    task_data: TaskCreateRequest,
    db: AsyncSession = Depends(get_session_dependency),
    task_engine: TaskExecutionEngine = Depends(get_task_engine)
) -> TaskResponse:
    """
    Create a new task for agent execution.
    
    Performance target: <100ms
    """
    current_user = get_current_user_from_request(request)
    
    try:
        # Create task record
        task = Task(
            id=str(uuid.uuid4()),
            title=task_data.title,
            description=task_data.description,
            type=task_data.type,
            priority=task_data.priority,
            status=TaskStatus.PENDING,
            parameters=task_data.parameters or {},
            requirements=task_data.requirements or {},
            metadata={
                "created_by": current_user.id,
                "created_at": datetime.utcnow().isoformat(),
                "version": "2.0"
            }
        )
        
        db.add(task)
        await db.commit()
        await db.refresh(task)
        
        # Submit task to execution engine
        await task_engine.submit_task(
            task_id=task.id,
            task_type=task.type,
            priority=task.priority,
            parameters=task.parameters
        )
        
        logger.info(
            "task_created",
            task_id=task.id,
            task_title=task.title,
            task_type=task.type.value,
            priority=task.priority.value,
            created_by=current_user.id
        )
        
        return TaskResponse.from_orm(task)
        
    except Exception as e:
        await db.rollback()
        logger.error("task_creation_failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create task: {str(e)}"
        )

@router.get("/", response_model=TaskListResponse)
async def list_tasks(
    request: Request,
    skip: int = Query(0, ge=0, description="Number of tasks to skip"),
    limit: int = Query(50, ge=1, le=1000, description="Number of tasks to return"),
    status: Optional[TaskStatus] = Query(None, description="Filter by task status"),
    type: Optional[TaskType] = Query(None, description="Filter by task type"),
    priority: Optional[TaskPriority] = Query(None, description="Filter by task priority"),
    assigned_agent_id: Optional[str] = Query(None, description="Filter by assigned agent"),
    db: AsyncSession = Depends(get_session_dependency)
) -> TaskListResponse:
    """
    List all tasks with optional filtering.
    
    Performance target: <100ms
    """
    try:
        # Build query with filters
        query = select(Task).options(selectinload(Task.assigned_agent))
        
        filters = []
        if status:
            filters.append(Task.status == status)
        if type:
            filters.append(Task.type == type)
        if priority:
            filters.append(Task.priority == priority)
        if assigned_agent_id:
            filters.append(Task.assigned_agent_id == assigned_agent_id)
            
        if filters:
            query = query.where(and_(*filters))
            
        # Apply pagination
        query = query.offset(skip).limit(limit)
        
        # Execute query
        result = await db.execute(query)
        tasks = result.scalars().all()
        
        # Get total count for pagination
        count_query = select(Task)
        if filters:
            count_query = count_query.where(and_(*filters))
            
        total_result = await db.execute(count_query)
        total = len(total_result.scalars().all())
        
        return TaskListResponse(
            tasks=[TaskResponse.from_orm(task) for task in tasks],
            total=total,
            skip=skip,
            limit=limit
        )
        
    except Exception as e:
        logger.error("task_list_failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list tasks: {str(e)}"
        )

@router.get("/{task_id}", response_model=TaskResponse)
async def get_task(
    task_id: str,
    db: AsyncSession = Depends(get_session_dependency)
) -> TaskResponse:
    """
    Get details of a specific task.
    
    Performance target: <100ms
    """
    try:
        # Query task with agent info
        query = select(Task).options(selectinload(Task.assigned_agent)).where(Task.id == task_id)
        result = await db.execute(query)
        task = result.scalar_one_or_none()
        
        if not task:
            raise HTTPException(
                status_code=404,
                detail=f"Task {task_id} not found"
            )
            
        return TaskResponse.from_orm(task)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("task_get_failed", task_id=task_id, error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get task: {str(e)}"
        )

@router.put("/{task_id}", response_model=TaskResponse)
async def update_task(
    request: Request,
    task_id: str,
    task_data: TaskUpdateRequest,
    db: AsyncSession = Depends(get_session_dependency),
    task_engine: TaskExecutionEngine = Depends(get_task_engine)
) -> TaskResponse:
    """
    Update an existing task.
    
    Performance target: <100ms
    """
    current_user = get_current_user_from_request(request)
    
    try:
        # Get existing task
        query = select(Task).where(Task.id == task_id)
        result = await db.execute(query)
        task = result.scalar_one_or_none()
        
        if not task:
            raise HTTPException(
                status_code=404,
                detail=f"Task {task_id} not found"
            )
        
        # Update task fields
        update_data = task_data.dict(exclude_unset=True)
        
        if update_data:
            update_data["updated_at"] = datetime.utcnow()
            update_data["updated_by"] = current_user.id
            
            # Update in database
            await db.execute(
                update(Task)
                .where(Task.id == task_id)
                .values(**update_data)
            )
            await db.commit()
            
            # Update in task engine if priority or parameters changed
            if "priority" in update_data or "parameters" in update_data:
                await task_engine.update_task(
                    task_id=task_id,
                    priority=update_data.get("priority", task.priority),
                    parameters=update_data.get("parameters", task.parameters)
                )
        
        # Get updated task
        result = await db.execute(query.options(selectinload(Task.assigned_agent)))
        updated_task = result.scalar_one()
        
        logger.info(
            "task_updated",
            task_id=task_id,
            updated_by=current_user.id,
            updated_fields=list(update_data.keys())
        )
        
        return TaskResponse.from_orm(updated_task)
        
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        logger.error("task_update_failed", task_id=task_id, error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update task: {str(e)}"
        )

@router.delete("/{task_id}", status_code=204)
async def delete_task(
    request: Request,
    task_id: str,
    db: AsyncSession = Depends(get_session_dependency),
    task_engine: TaskExecutionEngine = Depends(get_task_engine)
):
    """
    Delete a task from the system.
    
    Performance target: <100ms
    """
    current_user = get_current_user_from_request(request)
    
    try:
        # Check if task exists
        query = select(Task).where(Task.id == task_id)
        result = await db.execute(query)
        task = result.scalar_one_or_none()
        
        if not task:
            raise HTTPException(
                status_code=404,
                detail=f"Task {task_id} not found"
            )
        
        # Check if task is running
        if task.status == TaskStatus.RUNNING:
            raise HTTPException(
                status_code=400,
                detail="Cannot delete a running task"
            )
        
        # Cancel task in execution engine
        await task_engine.cancel_task(task_id)
        
        # Delete from database
        await db.execute(delete(Task).where(Task.id == task_id))
        await db.commit()
        
        logger.info(
            "task_deleted",
            task_id=task_id,
            deleted_by=current_user.id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        logger.error("task_delete_failed", task_id=task_id, error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete task: {str(e)}"
        )

@router.post("/{task_id}/assign", response_model=TaskResponse)
async def assign_task(
    request: Request,
    task_id: str,
    assignment_data: TaskAssignmentRequest,
    db: AsyncSession = Depends(get_session_dependency),
    task_engine: TaskExecutionEngine = Depends(get_task_engine)
) -> TaskResponse:
    """
    Assign a task to a specific agent.
    
    Performance target: <100ms
    """
    current_user = get_current_user_from_request(request)
    
    try:
        # Get task and agent
        task_query = select(Task).where(Task.id == task_id)
        task_result = await db.execute(task_query)
        task = task_result.scalar_one_or_none()
        
        if not task:
            raise HTTPException(
                status_code=404,
                detail=f"Task {task_id} not found"
            )
        
        agent_query = select(Agent).where(Agent.id == assignment_data.agent_id)
        agent_result = await db.execute(agent_query)
        agent = agent_result.scalar_one_or_none()
        
        if not agent:
            raise HTTPException(
                status_code=404,
                detail=f"Agent {assignment_data.agent_id} not found"
            )
        
        # Check if agent is available
        if agent.status != AgentStatus.ACTIVE:
            raise HTTPException(
                status_code=400,
                detail=f"Agent {assignment_data.agent_id} is not active"
            )
        
        # Check if task is assignable
        if task.status not in [TaskStatus.PENDING, TaskStatus.FAILED]:
            raise HTTPException(
                status_code=400,
                detail=f"Task {task_id} cannot be assigned (status: {task.status.value})"
            )
        
        # Assign task in execution engine
        await task_engine.assign_task(
            task_id=task_id,
            agent_id=assignment_data.agent_id
        )
        
        # Update task in database
        await db.execute(
            update(Task)
            .where(Task.id == task_id)
            .values(
                assigned_agent_id=assignment_data.agent_id,
                status=TaskStatus.ASSIGNED,
                updated_at=datetime.utcnow(),
                updated_by=current_user.id
            )
        )
        await db.commit()
        
        # Get updated task
        result = await db.execute(task_query.options(selectinload(Task.assigned_agent)))
        assigned_task = result.scalar_one()
        
        logger.info(
            "task_assigned",
            task_id=task_id,
            agent_id=assignment_data.agent_id,
            assigned_by=current_user.id
        )
        
        return TaskResponse.from_orm(assigned_task)
        
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        logger.error("task_assignment_failed", task_id=task_id, error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to assign task: {str(e)}"
        )

@router.post("/{task_id}/start")
async def start_task(
    request: Request,
    task_id: str,
    db: AsyncSession = Depends(get_session_dependency),
    task_engine: TaskExecutionEngine = Depends(get_task_engine)
):
    """
    Start execution of an assigned task.
    
    Performance target: <100ms
    """
    current_user = get_current_user_from_request(request)
    
    try:
        # Get task
        query = select(Task).where(Task.id == task_id)
        result = await db.execute(query)
        task = result.scalar_one_or_none()
        
        if not task:
            raise HTTPException(
                status_code=404,
                detail=f"Task {task_id} not found"
            )
        
        if task.status != TaskStatus.ASSIGNED:
            raise HTTPException(
                status_code=400,
                detail=f"Task must be assigned before starting (current status: {task.status.value})"
            )
        
        # Start task in execution engine
        await task_engine.start_task(task_id)
        
        # Update task status
        await db.execute(
            update(Task)
            .where(Task.id == task_id)
            .values(
                status=TaskStatus.RUNNING,
                started_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                updated_by=current_user.id
            )
        )
        await db.commit()
        
        logger.info(
            "task_started",
            task_id=task_id,
            started_by=current_user.id
        )
        
        return {
            "task_id": task_id,
            "status": "started",
            "message": "Task execution started"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        logger.error("task_start_failed", task_id=task_id, error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start task: {str(e)}"
        )

@router.post("/{task_id}/cancel")
async def cancel_task(
    request: Request,
    task_id: str,
    db: AsyncSession = Depends(get_session_dependency),
    task_engine: TaskExecutionEngine = Depends(get_task_engine)
):
    """
    Cancel a running task.
    
    Performance target: <100ms
    """
    current_user = get_current_user_from_request(request)
    
    try:
        # Get task
        query = select(Task).where(Task.id == task_id)
        result = await db.execute(query)
        task = result.scalar_one_or_none()
        
        if not task:
            raise HTTPException(
                status_code=404,
                detail=f"Task {task_id} not found"
            )
        
        if task.status not in [TaskStatus.RUNNING, TaskStatus.ASSIGNED]:
            raise HTTPException(
                status_code=400,
                detail=f"Can only cancel running or assigned tasks (current status: {task.status.value})"
            )
        
        # Cancel task in execution engine
        await task_engine.cancel_task(task_id)
        
        # Update task status
        await db.execute(
            update(Task)
            .where(Task.id == task_id)
            .values(
                status=TaskStatus.CANCELLED,
                updated_at=datetime.utcnow(),
                updated_by=current_user.id
            )
        )
        await db.commit()
        
        logger.info(
            "task_cancelled",
            task_id=task_id,
            cancelled_by=current_user.id
        )
        
        return {
            "task_id": task_id,
            "status": "cancelled",
            "message": "Task execution cancelled"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        logger.error("task_cancel_failed", task_id=task_id, error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to cancel task: {str(e)}"
        )

@router.get("/stats/overview", response_model=TaskStatsResponse)
async def get_task_stats(
    db: AsyncSession = Depends(get_session_dependency)
) -> TaskStatsResponse:
    """
    Get system-wide task statistics.
    
    Performance target: <100ms
    """
    try:
        # Get all tasks
        query = select(Task)
        result = await db.execute(query)
        tasks = result.scalars().all()
        
        # Calculate statistics
        total_tasks = len(tasks)
        
        status_counts = {}
        for status in TaskStatus:
            status_counts[status.value] = len([t for t in tasks if t.status == status])
        
        type_counts = {}
        for task_type in TaskType:
            type_counts[task_type.value] = len([t for t in tasks if t.type == task_type])
        
        priority_counts = {}
        for priority in TaskPriority:
            priority_counts[priority.value] = len([t for t in tasks if t.priority == priority])
        
        # Calculate completion rate
        completed = status_counts.get(TaskStatus.COMPLETED.value, 0)
        completion_rate = (completed / total_tasks * 100) if total_tasks > 0 else 0
        
        return TaskStatsResponse(
            total_tasks=total_tasks,
            status_breakdown=status_counts,
            type_breakdown=type_counts,
            priority_breakdown=priority_counts,
            completion_rate=round(completion_rate, 2),
            average_execution_time_minutes=0  # Would calculate from execution logs
        )
        
    except Exception as e:
        logger.error("task_stats_failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get task stats: {str(e)}"
        )