"""
Workflow management API endpoints for LeanVibe Agent Hive 2.0

Provides CRUD operations for managing multi-agent workflows with task coordination,
dependency management, execution monitoring, and progress tracking.
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
from ...models.workflow import Workflow, WorkflowStatus, WorkflowPriority
from ...models.task import Task, TaskStatus
from ...schemas.workflow import (
    WorkflowCreate, WorkflowUpdate, WorkflowResponse, WorkflowListResponse,
    WorkflowTaskAssignment, WorkflowExecutionRequest, WorkflowProgressResponse,
    WorkflowValidationResponse, WorkflowStatsResponse
)

logger = structlog.get_logger()
router = APIRouter()


@router.post("/", response_model=WorkflowResponse, status_code=201)
async def create_workflow(
    workflow_data: WorkflowCreate,
    db: AsyncSession = Depends(get_session_dependency)
) -> WorkflowResponse:
    """Create a new multi-agent workflow."""
    
    try:
        # Create workflow in database
        workflow = Workflow(
            name=workflow_data.name,
            description=workflow_data.description,
            priority=workflow_data.priority,
            definition=workflow_data.definition,
            context=workflow_data.context or {},
            variables=workflow_data.variables or {},
            estimated_duration=workflow_data.estimated_duration,
            due_date=workflow_data.due_date
        )
        
        db.add(workflow)
        await db.commit()
        await db.refresh(workflow)
        
        logger.info(
            "Workflow created",
            workflow_id=str(workflow.id),
            name=workflow.name,
            priority=workflow.priority.value
        )
        
        return WorkflowResponse.from_orm(workflow)
        
    except Exception as e:
        logger.error("Failed to create workflow", error=str(e))
        await db.rollback()
        raise HTTPException(status_code=500, detail="Failed to create workflow")


@router.get("/", response_model=WorkflowListResponse)
async def list_workflows(
    status: Optional[WorkflowStatus] = Query(None, description="Filter by workflow status"),
    priority: Optional[WorkflowPriority] = Query(None, description="Filter by workflow priority"),
    limit: int = Query(50, ge=1, le=100, description="Number of workflows to return"),
    offset: int = Query(0, ge=0, description="Number of workflows to skip"),
    db: AsyncSession = Depends(get_session_dependency)
) -> WorkflowListResponse:
    """List all workflows with optional filtering."""
    
    try:
        # Build query with filters
        query = select(Workflow)
        
        if status:
            query = query.where(Workflow.status == status)
        if priority:
            query = query.where(Workflow.priority == priority)
        
        query = query.offset(offset).limit(limit).order_by(Workflow.created_at.desc())
        
        result = await db.execute(query)
        workflows = result.scalars().all()
        
        # Get total count for pagination
        count_query = select(func.count(Workflow.id))
        if status:
            count_query = count_query.where(Workflow.status == status)
        if priority:
            count_query = count_query.where(Workflow.priority == priority)
        
        count_result = await db.execute(count_query)
        total = count_result.scalar()
        
        return WorkflowListResponse(
            workflows=[WorkflowResponse.from_orm(workflow) for workflow in workflows],
            total=total,
            offset=offset,
            limit=limit
        )
        
    except Exception as e:
        logger.error("Failed to list workflows", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve workflows")


@router.get("/{workflow_id}", response_model=WorkflowResponse)
async def get_workflow(
    workflow_id: uuid.UUID,
    db: AsyncSession = Depends(get_session_dependency)
) -> WorkflowResponse:
    """Get a specific workflow by ID."""
    
    try:
        result = await db.execute(
            select(Workflow).where(Workflow.id == workflow_id)
        )
        workflow = result.scalar_one_or_none()
        
        if not workflow:
            raise HTTPException(status_code=404, detail="Workflow not found")
        
        return WorkflowResponse.from_orm(workflow)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get workflow", workflow_id=str(workflow_id), error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve workflow")


@router.put("/{workflow_id}", response_model=WorkflowResponse)
async def update_workflow(
    workflow_id: uuid.UUID,
    workflow_data: WorkflowUpdate,
    db: AsyncSession = Depends(get_session_dependency)
) -> WorkflowResponse:
    """Update an existing workflow."""
    
    try:
        # Check if workflow exists
        result = await db.execute(
            select(Workflow).where(Workflow.id == workflow_id)
        )
        workflow = result.scalar_one_or_none()
        
        if not workflow:
            raise HTTPException(status_code=404, detail="Workflow not found")
        
        # Update workflow fields
        update_data = workflow_data.dict(exclude_unset=True)
        update_data["updated_at"] = datetime.utcnow()
        
        await db.execute(
            update(Workflow).where(Workflow.id == workflow_id).values(**update_data)
        )
        await db.commit()
        
        # Fetch updated workflow
        result = await db.execute(
            select(Workflow).where(Workflow.id == workflow_id)
        )
        updated_workflow = result.scalar_one()
        
        logger.info(
            "Workflow updated",
            workflow_id=str(workflow_id),
            updated_fields=list(update_data.keys())
        )
        
        return WorkflowResponse.from_orm(updated_workflow)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to update workflow", workflow_id=str(workflow_id), error=str(e))
        await db.rollback()
        raise HTTPException(status_code=500, detail="Failed to update workflow")


@router.delete("/{workflow_id}", status_code=204)
async def delete_workflow(
    workflow_id: uuid.UUID,
    db: AsyncSession = Depends(get_session_dependency)
) -> None:
    """Cancel a workflow (soft delete)."""
    
    try:
        # Check if workflow exists
        result = await db.execute(
            select(Workflow).where(Workflow.id == workflow_id)
        )
        workflow = result.scalar_one_or_none()
        
        if not workflow:
            raise HTTPException(status_code=404, detail="Workflow not found")
        
        # Soft delete by setting status to cancelled
        await db.execute(
            update(Workflow)
            .where(Workflow.id == workflow_id)
            .values(
                status=WorkflowStatus.CANCELLED,
                completed_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
        )
        await db.commit()
        
        logger.info("Workflow cancelled", workflow_id=str(workflow_id))
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to cancel workflow", workflow_id=str(workflow_id), error=str(e))
        await db.rollback()
        raise HTTPException(status_code=500, detail="Failed to cancel workflow")


@router.post("/{workflow_id}/tasks", response_model=WorkflowResponse)
async def add_task_to_workflow(
    workflow_id: uuid.UUID,
    task_assignment: WorkflowTaskAssignment,
    db: AsyncSession = Depends(get_session_dependency)
) -> WorkflowResponse:
    """Add a task to a workflow with optional dependencies."""
    
    try:
        # Check if workflow exists
        workflow_result = await db.execute(
            select(Workflow).where(Workflow.id == workflow_id)
        )
        workflow = workflow_result.scalar_one_or_none()
        
        if not workflow:
            raise HTTPException(status_code=404, detail="Workflow not found")
        
        # Check if task exists
        task_result = await db.execute(
            select(Task).where(Task.id == task_assignment.task_id)
        )
        task = task_result.scalar_one_or_none()
        
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        
        # Validate dependencies exist
        if task_assignment.dependencies:
            deps_result = await db.execute(
                select(Task.id).where(Task.id.in_(task_assignment.dependencies))
            )
            existing_deps = [dep[0] for dep in deps_result.all()]
            
            if len(existing_deps) != len(task_assignment.dependencies):
                missing_deps = set(task_assignment.dependencies) - set(existing_deps)
                raise HTTPException(
                    status_code=400,
                    detail=f"Dependency tasks not found: {missing_deps}"
                )
        
        # Add task to workflow
        task_ids = list(workflow.task_ids) if workflow.task_ids else []
        if task_assignment.task_id not in task_ids:
            task_ids.append(task_assignment.task_id)
        
        # Update dependencies
        dependencies = workflow.dependencies or {}
        if task_assignment.dependencies:
            dependencies[str(task_assignment.task_id)] = [str(dep) for dep in task_assignment.dependencies]
        
        # Update workflow
        await db.execute(
            update(Workflow)
            .where(Workflow.id == workflow_id)
            .values(
                task_ids=task_ids,
                dependencies=dependencies,
                total_tasks=len(task_ids),
                updated_at=datetime.utcnow()
            )
        )
        await db.commit()
        
        # Fetch updated workflow
        result = await db.execute(
            select(Workflow).where(Workflow.id == workflow_id)
        )
        updated_workflow = result.scalar_one()
        
        logger.info(
            "Task added to workflow",
            workflow_id=str(workflow_id),
            task_id=str(task_assignment.task_id),
            dependencies=len(task_assignment.dependencies or [])
        )
        
        return WorkflowResponse.from_orm(updated_workflow)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to add task to workflow",
            workflow_id=str(workflow_id),
            task_id=str(task_assignment.task_id),
            error=str(e)
        )
        await db.rollback()
        raise HTTPException(status_code=500, detail="Failed to add task to workflow")


@router.delete("/{workflow_id}/tasks/{task_id}", response_model=WorkflowResponse)
async def remove_task_from_workflow(
    workflow_id: uuid.UUID,
    task_id: uuid.UUID,
    db: AsyncSession = Depends(get_session_dependency)
) -> WorkflowResponse:
    """Remove a task from a workflow."""
    
    try:
        # Check if workflow exists
        result = await db.execute(
            select(Workflow).where(Workflow.id == workflow_id)
        )
        workflow = result.scalar_one_or_none()
        
        if not workflow:
            raise HTTPException(status_code=404, detail="Workflow not found")
        
        # Remove task from workflow
        task_ids = list(workflow.task_ids) if workflow.task_ids else []
        if task_id in task_ids:
            task_ids.remove(task_id)
        
        # Remove from dependencies
        dependencies = workflow.dependencies or {}
        dependencies.pop(str(task_id), None)
        
        # Remove as dependency from other tasks
        for task, deps in dependencies.items():
            if str(task_id) in deps:
                deps.remove(str(task_id))
        
        # Update workflow
        await db.execute(
            update(Workflow)
            .where(Workflow.id == workflow_id)
            .values(
                task_ids=task_ids,
                dependencies=dependencies,
                total_tasks=len(task_ids),
                updated_at=datetime.utcnow()
            )
        )
        await db.commit()
        
        # Fetch updated workflow
        result = await db.execute(
            select(Workflow).where(Workflow.id == workflow_id)
        )
        updated_workflow = result.scalar_one()
        
        logger.info(
            "Task removed from workflow",
            workflow_id=str(workflow_id),
            task_id=str(task_id)
        )
        
        return WorkflowResponse.from_orm(updated_workflow)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to remove task from workflow",
            workflow_id=str(workflow_id),
            task_id=str(task_id),
            error=str(e)
        )
        await db.rollback()
        raise HTTPException(status_code=500, detail="Failed to remove task from workflow")


@router.post("/{workflow_id}/execute", response_model=WorkflowResponse)
async def execute_workflow(
    workflow_id: uuid.UUID,
    execution_request: WorkflowExecutionRequest = None,
    db: AsyncSession = Depends(get_session_dependency)
) -> WorkflowResponse:
    """Start execution of a workflow."""
    
    try:
        # Check if workflow exists
        result = await db.execute(
            select(Workflow).where(Workflow.id == workflow_id)
        )
        workflow = result.scalar_one_or_none()
        
        if not workflow:
            raise HTTPException(status_code=404, detail="Workflow not found")
        
        if workflow.status not in [WorkflowStatus.CREATED, WorkflowStatus.READY, WorkflowStatus.PAUSED]:
            raise HTTPException(
                status_code=400,
                detail=f"Workflow cannot be executed in status: {workflow.status.value}"
            )
        
        # Validate workflow dependencies
        errors = workflow.validate_dependencies()
        if errors:
            raise HTTPException(
                status_code=400,
                detail=f"Workflow validation failed: {'; '.join(errors)}"
            )
        
        # Update context and variables if provided
        update_values = {
            "status": WorkflowStatus.RUNNING,
            "started_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        
        if execution_request:
            if execution_request.context_override:
                update_values["context"] = execution_request.context_override
            if execution_request.variables_override:
                update_values["variables"] = execution_request.variables_override
        
        await db.execute(
            update(Workflow).where(Workflow.id == workflow_id).values(**update_values)
        )
        await db.commit()
        
        # Fetch updated workflow
        result = await db.execute(
            select(Workflow).where(Workflow.id == workflow_id)
        )
        updated_workflow = result.scalar_one()
        
        logger.info("Workflow execution started", workflow_id=str(workflow_id))
        
        return WorkflowResponse.from_orm(updated_workflow)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to execute workflow", workflow_id=str(workflow_id), error=str(e))
        await db.rollback()
        raise HTTPException(status_code=500, detail="Failed to execute workflow")


@router.post("/{workflow_id}/pause", response_model=WorkflowResponse)
async def pause_workflow(
    workflow_id: uuid.UUID,
    db: AsyncSession = Depends(get_session_dependency)
) -> WorkflowResponse:
    """Pause a running workflow."""
    
    try:
        # Check if workflow exists and can be paused
        result = await db.execute(
            select(Workflow).where(Workflow.id == workflow_id)
        )
        workflow = result.scalar_one_or_none()
        
        if not workflow:
            raise HTTPException(status_code=404, detail="Workflow not found")
        
        if workflow.status != WorkflowStatus.RUNNING:
            raise HTTPException(
                status_code=400,
                detail=f"Workflow cannot be paused in status: {workflow.status.value}"
            )
        
        # Pause workflow
        await db.execute(
            update(Workflow)
            .where(Workflow.id == workflow_id)
            .values(status=WorkflowStatus.PAUSED, updated_at=datetime.utcnow())
        )
        await db.commit()
        
        # Fetch updated workflow
        result = await db.execute(
            select(Workflow).where(Workflow.id == workflow_id)
        )
        updated_workflow = result.scalar_one()
        
        logger.info("Workflow paused", workflow_id=str(workflow_id))
        
        return WorkflowResponse.from_orm(updated_workflow)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to pause workflow", workflow_id=str(workflow_id), error=str(e))
        await db.rollback()
        raise HTTPException(status_code=500, detail="Failed to pause workflow")


@router.post("/{workflow_id}/resume", response_model=WorkflowResponse)
async def resume_workflow(
    workflow_id: uuid.UUID,
    db: AsyncSession = Depends(get_session_dependency)
) -> WorkflowResponse:
    """Resume a paused workflow."""
    
    try:
        # Check if workflow exists and can be resumed
        result = await db.execute(
            select(Workflow).where(Workflow.id == workflow_id)
        )
        workflow = result.scalar_one_or_none()
        
        if not workflow:
            raise HTTPException(status_code=404, detail="Workflow not found")
        
        if workflow.status != WorkflowStatus.PAUSED:
            raise HTTPException(
                status_code=400,
                detail=f"Workflow cannot be resumed in status: {workflow.status.value}"
            )
        
        # Resume workflow
        await db.execute(
            update(Workflow)
            .where(Workflow.id == workflow_id)
            .values(status=WorkflowStatus.RUNNING, updated_at=datetime.utcnow())
        )
        await db.commit()
        
        # Fetch updated workflow
        result = await db.execute(
            select(Workflow).where(Workflow.id == workflow_id)
        )
        updated_workflow = result.scalar_one()
        
        logger.info("Workflow resumed", workflow_id=str(workflow_id))
        
        return WorkflowResponse.from_orm(updated_workflow)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to resume workflow", workflow_id=str(workflow_id), error=str(e))
        await db.rollback()
        raise HTTPException(status_code=500, detail="Failed to resume workflow")


@router.get("/{workflow_id}/progress", response_model=WorkflowProgressResponse)
async def get_workflow_progress(
    workflow_id: uuid.UUID,
    db: AsyncSession = Depends(get_session_dependency)
) -> WorkflowProgressResponse:
    """Get detailed progress information for a workflow."""
    
    try:
        # Get workflow
        result = await db.execute(
            select(Workflow).where(Workflow.id == workflow_id)
        )
        workflow = result.scalar_one_or_none()
        
        if not workflow:
            raise HTTPException(status_code=404, detail="Workflow not found")
        
        # Get task status counts
        if workflow.task_ids:
            tasks_result = await db.execute(
                select(Task.id, Task.status).where(Task.id.in_(workflow.task_ids))
            )
            task_statuses = {str(task_id): status for task_id, status in tasks_result.all()}
            
            completed_task_ids = [
                task_id for task_id, status in task_statuses.items() 
                if status == TaskStatus.COMPLETED
            ]
            
            failed_task_ids = [
                task_id for task_id, status in task_statuses.items()
                if status == TaskStatus.FAILED
            ]
            
            in_progress_task_ids = [
                task_id for task_id, status in task_statuses.items()
                if status == TaskStatus.IN_PROGRESS
            ]
            
            # Get ready tasks based on dependencies
            ready_task_ids = workflow.get_ready_tasks(completed_task_ids)
            
        else:
            completed_task_ids = []
            failed_task_ids = []
            in_progress_task_ids = []
            ready_task_ids = []
        
        pending_tasks = workflow.total_tasks - workflow.completed_tasks - workflow.failed_tasks
        
        return WorkflowProgressResponse(
            workflow_id=workflow.id,
            name=workflow.name,
            status=workflow.status,
            completion_percentage=workflow.get_completion_percentage(),
            total_tasks=workflow.total_tasks,
            completed_tasks=len(completed_task_ids),
            failed_tasks=len(failed_task_ids),
            pending_tasks=max(0, pending_tasks),
            estimated_completion=workflow.estimate_completion_time(),
            current_tasks=[uuid.UUID(task_id) for task_id in in_progress_task_ids],
            ready_tasks=[uuid.UUID(task_id) for task_id in ready_task_ids]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get workflow progress", workflow_id=str(workflow_id), error=str(e))
        raise HTTPException(status_code=500, detail="Failed to get workflow progress")


@router.get("/{workflow_id}/validate", response_model=WorkflowValidationResponse)
async def validate_workflow(
    workflow_id: uuid.UUID,
    db: AsyncSession = Depends(get_session_dependency)
) -> WorkflowValidationResponse:
    """Validate a workflow for execution readiness."""
    
    try:
        # Get workflow
        result = await db.execute(
            select(Workflow).where(Workflow.id == workflow_id)
        )
        workflow = result.scalar_one_or_none()
        
        if not workflow:
            raise HTTPException(status_code=404, detail="Workflow not found")
        
        errors = []
        warnings = []
        
        # Validate dependencies
        dependency_errors = workflow.validate_dependencies()
        errors.extend(dependency_errors)
        
        # Check if all tasks exist
        if workflow.task_ids:
            tasks_result = await db.execute(
                select(Task.id).where(Task.id.in_(workflow.task_ids))
            )
            existing_task_ids = [str(task_id[0]) for task_id in tasks_result.all()]
            missing_tasks = set(str(task_id) for task_id in workflow.task_ids) - set(existing_task_ids)
            
            if missing_tasks:
                errors.append(f"Missing tasks: {missing_tasks}")
        
        # Check for orphaned tasks (no path to completion)
        if workflow.dependencies and workflow.task_ids:
            task_ids_str = [str(task_id) for task_id in workflow.task_ids]
            has_dependencies = set(workflow.dependencies.keys())
            no_dependencies = set(task_ids_str) - has_dependencies
            
            if len(no_dependencies) == 0 and len(task_ids_str) > 1:
                warnings.append("No tasks without dependencies - workflow may not be executable")
        
        # Check for unreachable tasks
        if workflow.total_tasks == 0:
            warnings.append("Workflow has no tasks")
        
        is_valid = len(errors) == 0
        
        return WorkflowValidationResponse(
            workflow_id=workflow.id,
            is_valid=is_valid,
            errors=errors,
            warnings=warnings
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to validate workflow", workflow_id=str(workflow_id), error=str(e))
        raise HTTPException(status_code=500, detail="Failed to validate workflow")


@router.get("/{workflow_id}/stats", response_model=WorkflowStatsResponse)
async def get_workflow_stats(
    workflow_id: uuid.UUID,
    db: AsyncSession = Depends(get_session_dependency)
) -> WorkflowStatsResponse:
    """Get detailed statistics for a workflow."""
    
    try:
        # Get workflow
        result = await db.execute(
            select(Workflow).where(Workflow.id == workflow_id)
        )
        workflow = result.scalar_one_or_none()
        
        if not workflow:
            raise HTTPException(status_code=404, detail="Workflow not found")
        
        # Calculate basic stats
        total_execution_time = workflow.actual_duration or 0
        
        # Get task performance data
        if workflow.task_ids:
            tasks_result = await db.execute(
                select(Task.id, Task.actual_effort, Task.status)
                .where(Task.id.in_(workflow.task_ids))
            )
            task_data = tasks_result.all()
            
            completed_tasks = [t for t in task_data if t.status == TaskStatus.COMPLETED]
            total_task_time = sum(t.actual_effort or 0 for t in completed_tasks)
            average_task_time = total_task_time / max(len(completed_tasks), 1)
            
            success_rate = (len(completed_tasks) / max(len(task_data), 1)) * 100
            
            # Calculate efficiency (actual vs estimated)
            efficiency_score = 1.0
            if workflow.estimated_duration and workflow.actual_duration:
                efficiency_score = min(1.0, workflow.estimated_duration / workflow.actual_duration)
        
        else:
            average_task_time = 0.0
            success_rate = 0.0
            efficiency_score = 1.0
        
        # TODO: Implement bottleneck detection and critical path analysis
        bottleneck_tasks = []
        critical_path_duration = total_execution_time
        
        return WorkflowStatsResponse(
            workflow_id=workflow.id,
            name=workflow.name,
            total_execution_time=total_execution_time,
            average_task_time=average_task_time,
            success_rate=success_rate,
            efficiency_score=efficiency_score,
            bottleneck_tasks=bottleneck_tasks,
            critical_path_duration=critical_path_duration
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get workflow stats", workflow_id=str(workflow_id), error=str(e))
        raise HTTPException(status_code=500, detail="Failed to get workflow stats")