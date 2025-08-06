"""
Task management API endpoints for LeanVibe Agent Hive 2.0

Provides CRUD operations for managing development tasks with assignment,
status tracking, delegation, and comprehensive lifecycle management.
"""

import uuid
from typing import List, Optional
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete, func
from sqlalchemy.orm import selectinload

import structlog

from ...core.database import get_session_dependency
from ...models.task import Task, TaskStatus, TaskPriority, TaskType
from ...models.agent import Agent, AgentStatus
from ...schemas.task import (
    TaskCreate, TaskUpdate, TaskResponse, TaskListResponse
)

logger = structlog.get_logger()
router = APIRouter()


@router.post("/", response_model=TaskResponse, status_code=201)
async def create_task(
    task_data: TaskCreate,
    db: AsyncSession = Depends(get_session_dependency)
) -> TaskResponse:
    """Create a new development task."""
    
    try:
        # Create task in database
        task = Task(
            title=task_data.title,
            description=task_data.description,
            task_type=task_data.task_type,
            priority=task_data.priority,
            required_capabilities=task_data.required_capabilities or [],
            estimated_effort=task_data.estimated_effort,
            context=task_data.context or {}
        )
        
        db.add(task)
        await db.commit()
        await db.refresh(task)
        
        logger.info(
            "Task created",
            task_id=str(task.id),
            title=task.title,
            priority=task.priority.value
        )
        
        return TaskResponse.from_orm(task)
        
    except Exception as e:
        logger.error("Failed to create task", error=str(e))
        await db.rollback()
        raise HTTPException(status_code=500, detail="Failed to create task")


@router.get("/", response_model=TaskListResponse)
async def list_tasks(
    status: Optional[TaskStatus] = Query(None, description="Filter by task status"),
    priority: Optional[TaskPriority] = Query(None, description="Filter by task priority"),
    task_type: Optional[TaskType] = Query(None, description="Filter by task type"),
    assigned_agent_id: Optional[uuid.UUID] = Query(None, description="Filter by assigned agent"),
    limit: int = Query(50, ge=1, le=100, description="Number of tasks to return"),
    offset: int = Query(0, ge=0, description="Number of tasks to skip"),
    db: AsyncSession = Depends(get_session_dependency)
) -> TaskListResponse:
    """List all tasks with optional filtering."""
    
    try:
        # Build query with filters
        query = select(Task)
        
        if status:
            query = query.where(Task.status == status)
        if priority:
            query = query.where(Task.priority == priority)
        if task_type:
            query = query.where(Task.task_type == task_type)
        if assigned_agent_id:
            query = query.where(Task.assigned_agent_id == assigned_agent_id)
        
        query = query.offset(offset).limit(limit).order_by(Task.created_at.desc())
        
        result = await db.execute(query)
        tasks = result.scalars().all()
        
        # Get total count for pagination
        count_query = select(func.count(Task.id))
        if status:
            count_query = count_query.where(Task.status == status)
        if priority:
            count_query = count_query.where(Task.priority == priority)
        if task_type:
            count_query = count_query.where(Task.task_type == task_type)
        if assigned_agent_id:
            count_query = count_query.where(Task.assigned_agent_id == assigned_agent_id)
        
        count_result = await db.execute(count_query)
        total = count_result.scalar()
        
        return TaskListResponse(
            tasks=[TaskResponse.from_orm(task) for task in tasks],
            total=total,
            offset=offset,
            limit=limit
        )
        
    except Exception as e:
        logger.error("Failed to list tasks", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve tasks")


@router.get("/{task_id}", response_model=TaskResponse)
async def get_task(
    task_id: uuid.UUID,
    db: AsyncSession = Depends(get_session_dependency)
) -> TaskResponse:
    """Get a specific task by ID."""
    
    try:
        result = await db.execute(
            select(Task).where(Task.id == task_id)
        )
        task = result.scalar_one_or_none()
        
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        
        return TaskResponse.from_orm(task)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get task", task_id=str(task_id), error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve task")


@router.put("/{task_id}", response_model=TaskResponse)
async def update_task(
    task_id: uuid.UUID,
    task_data: TaskUpdate,
    db: AsyncSession = Depends(get_session_dependency)
) -> TaskResponse:
    """Update an existing task."""
    
    try:
        # Check if task exists
        result = await db.execute(
            select(Task).where(Task.id == task_id)
        )
        task = result.scalar_one_or_none()
        
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        
        # Update task fields
        update_data = task_data.dict(exclude_unset=True)
        update_data["updated_at"] = datetime.utcnow()
        
        await db.execute(
            update(Task).where(Task.id == task_id).values(**update_data)
        )
        await db.commit()
        
        # Fetch updated task
        result = await db.execute(
            select(Task).where(Task.id == task_id)
        )
        updated_task = result.scalar_one()
        
        logger.info(
            "Task updated",
            task_id=str(task_id),
            updated_fields=list(update_data.keys())
        )
        
        return TaskResponse.from_orm(updated_task)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to update task", task_id=str(task_id), error=str(e))
        await db.rollback()
        raise HTTPException(status_code=500, detail="Failed to update task")


@router.delete("/{task_id}", status_code=204)
async def delete_task(
    task_id: uuid.UUID,
    db: AsyncSession = Depends(get_session_dependency)
) -> None:
    """Cancel a task (soft delete)."""
    
    try:
        # Check if task exists
        result = await db.execute(
            select(Task).where(Task.id == task_id)
        )
        task = result.scalar_one_or_none()
        
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        
        # Soft delete by setting status to cancelled
        await db.execute(
            update(Task)
            .where(Task.id == task_id)
            .values(
                status=TaskStatus.CANCELLED,
                completed_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
        )
        await db.commit()
        
        logger.info("Task cancelled", task_id=str(task_id))
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to cancel task", task_id=str(task_id), error=str(e))
        await db.rollback()
        raise HTTPException(status_code=500, detail="Failed to cancel task")


@router.post("/{task_id}/assign/{agent_id}", response_model=TaskResponse)
async def assign_task_to_agent(
    task_id: uuid.UUID,
    agent_id: uuid.UUID,
    db: AsyncSession = Depends(get_session_dependency)
) -> TaskResponse:
    """Assign a task to a specific agent."""
    
    try:
        # Check if task exists and is available for assignment
        task_result = await db.execute(
            select(Task).where(Task.id == task_id)
        )
        task = task_result.scalar_one_or_none()
        
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        
        if task.status not in [TaskStatus.PENDING]:
            raise HTTPException(
                status_code=400, 
                detail=f"Task cannot be assigned in status: {task.status.value}"
            )
        
        # Check if agent exists and is available
        agent_result = await db.execute(
            select(Agent).where(Agent.id == agent_id)
        )
        agent = agent_result.scalar_one_or_none()
        
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        if agent.status != AgentStatus.active:
            raise HTTPException(
                status_code=400,
                detail=f"Agent is not available (status: {agent.status.value})"
            )
        
        # Assign task to agent
        await db.execute(
            update(Task)
            .where(Task.id == task_id)
            .values(
                assigned_agent_id=agent_id,
                status=TaskStatus.ASSIGNED,
                assigned_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
        )
        await db.commit()
        
        # Fetch updated task
        result = await db.execute(
            select(Task).where(Task.id == task_id)
        )
        updated_task = result.scalar_one()
        
        logger.info(
            "Task assigned to agent",
            task_id=str(task_id),
            agent_id=str(agent_id)
        )
        
        return TaskResponse.from_orm(updated_task)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to assign task",
            task_id=str(task_id),
            agent_id=str(agent_id),
            error=str(e)
        )
        await db.rollback()
        raise HTTPException(status_code=500, detail="Failed to assign task")


@router.post("/{task_id}/start", response_model=TaskResponse)
async def start_task(
    task_id: uuid.UUID,
    db: AsyncSession = Depends(get_session_dependency)
) -> TaskResponse:
    """Start execution of an assigned task."""
    
    try:
        # Check if task exists and can be started
        result = await db.execute(
            select(Task).where(Task.id == task_id)
        )
        task = result.scalar_one_or_none()
        
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        
        if task.status != TaskStatus.ASSIGNED:
            raise HTTPException(
                status_code=400,
                detail=f"Task cannot be started in status: {task.status.value}"
            )
        
        # Start task execution
        await db.execute(
            update(Task)
            .where(Task.id == task_id)
            .values(
                status=TaskStatus.IN_PROGRESS,
                started_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
        )
        await db.commit()
        
        # Fetch updated task
        result = await db.execute(
            select(Task).where(Task.id == task_id)
        )
        updated_task = result.scalar_one()
        
        logger.info("Task started", task_id=str(task_id))
        
        return TaskResponse.from_orm(updated_task)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to start task", task_id=str(task_id), error=str(e))
        await db.rollback()
        raise HTTPException(status_code=500, detail="Failed to start task")


@router.post("/{task_id}/complete", response_model=TaskResponse)
async def complete_task(
    task_id: uuid.UUID,
    result: dict,
    db: AsyncSession = Depends(get_session_dependency)
) -> TaskResponse:
    """Mark a task as completed with result data."""
    
    try:
        # Check if task exists and can be completed
        task_result = await db.execute(
            select(Task).where(Task.id == task_id)
        )
        task = task_result.scalar_one_or_none()
        
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        
        if task.status != TaskStatus.IN_PROGRESS:
            raise HTTPException(
                status_code=400,
                detail=f"Task cannot be completed in status: {task.status.value}"
            )
        
        # Calculate actual effort if started_at is available
        actual_effort = None
        if task.started_at:
            duration = datetime.utcnow() - task.started_at
            actual_effort = int(duration.total_seconds() / 60)
        
        # Complete task
        await db.execute(
            update(Task)
            .where(Task.id == task_id)
            .values(
                status=TaskStatus.COMPLETED,
                result=result,
                actual_effort=actual_effort,
                completed_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
        )
        await db.commit()
        
        # Fetch updated task
        task_result = await db.execute(
            select(Task).where(Task.id == task_id)
        )
        updated_task = task_result.scalar_one()
        
        logger.info(
            "Task completed",
            task_id=str(task_id),
            actual_effort=actual_effort
        )
        
        return TaskResponse.from_orm(updated_task)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to complete task", task_id=str(task_id), error=str(e))
        await db.rollback()
        raise HTTPException(status_code=500, detail="Failed to complete task")


@router.post("/{task_id}/fail", response_model=TaskResponse)
async def fail_task(
    task_id: uuid.UUID,
    error_message: str,
    can_retry: bool = True,
    db: AsyncSession = Depends(get_session_dependency)
) -> TaskResponse:
    """Mark a task as failed with error information."""
    
    try:
        # Check if task exists
        result = await db.execute(
            select(Task).where(Task.id == task_id)
        )
        task = result.scalar_one_or_none()
        
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        
        # Calculate retry logic
        new_retry_count = task.retry_count + 1
        new_status = TaskStatus.PENDING if (can_retry and new_retry_count < task.max_retries) else TaskStatus.FAILED
        
        update_values = {
            "error_message": error_message,
            "retry_count": new_retry_count,
            "status": new_status,
            "updated_at": datetime.utcnow()
        }
        
        # If finally failed, set completion timestamp
        if new_status == TaskStatus.FAILED:
            update_values["completed_at"] = datetime.utcnow()
            
            # Calculate actual effort if started_at is available
            if task.started_at:
                duration = datetime.utcnow() - task.started_at
                update_values["actual_effort"] = int(duration.total_seconds() / 60)
        
        await db.execute(
            update(Task).where(Task.id == task_id).values(**update_values)
        )
        await db.commit()
        
        # Fetch updated task
        result = await db.execute(
            select(Task).where(Task.id == task_id)
        )
        updated_task = result.scalar_one()
        
        logger.info(
            "Task failed",
            task_id=str(task_id),
            retry_count=new_retry_count,
            final_failure=new_status == TaskStatus.FAILED
        )
        
        return TaskResponse.from_orm(updated_task)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to mark task as failed", task_id=str(task_id), error=str(e))
        await db.rollback()
        raise HTTPException(status_code=500, detail="Failed to mark task as failed")


@router.get("/agent/{agent_id}", response_model=TaskListResponse)
async def get_agent_tasks(
    agent_id: uuid.UUID,
    status: Optional[TaskStatus] = Query(None, description="Filter by task status"),
    limit: int = Query(50, ge=1, le=100, description="Number of tasks to return"),
    offset: int = Query(0, ge=0, description="Number of tasks to skip"),
    db: AsyncSession = Depends(get_session_dependency)
) -> TaskListResponse:
    """Get all tasks assigned to a specific agent."""
    
    try:
        # Build query for agent's tasks
        query = select(Task).where(Task.assigned_agent_id == agent_id)
        
        if status:
            query = query.where(Task.status == status)
        
        query = query.offset(offset).limit(limit).order_by(Task.created_at.desc())
        
        result = await db.execute(query)
        tasks = result.scalars().all()
        
        # Get total count
        count_query = select(func.count(Task.id)).where(Task.assigned_agent_id == agent_id)
        if status:
            count_query = count_query.where(Task.status == status)
        
        count_result = await db.execute(count_query)
        total = count_result.scalar()
        
        return TaskListResponse(
            tasks=[TaskResponse.from_orm(task) for task in tasks],
            total=total,
            offset=offset,
            limit=limit
        )
        
    except Exception as e:
        logger.error("Failed to get agent tasks", agent_id=str(agent_id), error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve agent tasks")