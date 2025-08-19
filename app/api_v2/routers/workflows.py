"""
Workflows API - Consolidated workflow management endpoints

Consolidates intelligent_scheduling.py, v1/workflows.py,
v1/automated_scheduler_vs7_2.py, and v1/coordination.py
into a unified RESTful resource for workflow orchestration.

Performance target: <150ms P95 response time
"""

import uuid
from typing import List, Optional, Dict, Any
from datetime import datetime

import structlog
from fastapi import APIRouter, Request, HTTPException, Query, BackgroundTasks
from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete, and_
from sqlalchemy.orm import selectinload

from ...core.database import get_session_dependency
from ...core.workflow_engine import WorkflowEngine
from ...models.workflow import Workflow, WorkflowStatus, WorkflowType
from ...models.task import Task, TaskStatus
from ...schemas.workflow import (
    WorkflowCreate,
    WorkflowUpdate,
    WorkflowResponse,
    WorkflowListResponse,
    WorkflowExecutionRequest,
    WorkflowStatsResponse
)
from ..middleware import (
    get_current_user_from_request
)

logger = structlog.get_logger()
router = APIRouter()

# Workflow engine dependency
async def get_workflow_engine() -> WorkflowEngine:
    """Get workflow engine instance."""
    return WorkflowEngine()

@router.post("/", response_model=WorkflowResponse, status_code=201)
async def create_workflow(
    request: Request,
    workflow_data: WorkflowCreate,
    db: AsyncSession = Depends(get_session_dependency),
    workflow_engine: WorkflowEngine = Depends(get_workflow_engine)
) -> WorkflowResponse:
    """
    Create a new workflow definition.
    
    Performance target: <150ms
    """
    current_user = get_current_user_from_request(request)
    
    try:
        # Create workflow record
        workflow = Workflow(
            id=str(uuid.uuid4()),
            name=workflow_data.name,
            description=workflow_data.description,
            type=workflow_data.type,
            definition=workflow_data.definition,
            status=WorkflowStatus.DRAFT,
            configuration=workflow_data.configuration or {},
            metadata={
                "created_by": current_user.id,
                "created_at": datetime.utcnow().isoformat(),
                "version": "2.0"
            }
        )
        
        db.add(workflow)
        await db.commit()
        await db.refresh(workflow)
        
        # Register workflow in engine
        await workflow_engine.register_workflow(
            workflow_id=workflow.id,
            definition=workflow.definition,
            configuration=workflow.configuration
        )
        
        logger.info(
            "workflow_created",
            workflow_id=workflow.id,
            workflow_name=workflow.name,
            workflow_type=workflow.type.value,
            created_by=current_user.id
        )
        
        return WorkflowResponse.from_orm(workflow)
        
    except Exception as e:
        await db.rollback()
        logger.error("workflow_creation_failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create workflow: {str(e)}"
        )

@router.get("/", response_model=WorkflowListResponse)
async def list_workflows(
    request: Request,
    skip: int = Query(0, ge=0, description="Number of workflows to skip"),
    limit: int = Query(50, ge=1, le=1000, description="Number of workflows to return"),
    status: Optional[WorkflowStatus] = Query(None, description="Filter by workflow status"),
    type: Optional[WorkflowType] = Query(None, description="Filter by workflow type"),
    db: AsyncSession = Depends(get_session_dependency)
) -> WorkflowListResponse:
    """
    List all workflows with optional filtering.
    
    Performance target: <150ms
    """
    try:
        # Build query with filters
        query = select(Workflow)
        
        if status:
            query = query.where(Workflow.status == status)
        if type:
            query = query.where(Workflow.type == type)
            
        # Apply pagination
        query = query.offset(skip).limit(limit)
        
        # Execute query
        result = await db.execute(query)
        workflows = result.scalars().all()
        
        # Get total count for pagination
        count_query = select(Workflow)
        if status:
            count_query = count_query.where(Workflow.status == status)
        if type:
            count_query = count_query.where(Workflow.type == type)
            
        total_result = await db.execute(count_query)
        total = len(total_result.scalars().all())
        
        return WorkflowListResponse(
            workflows=[WorkflowResponse.from_orm(workflow) for workflow in workflows],
            total=total,
            skip=skip,
            limit=limit
        )
        
    except Exception as e:
        logger.error("workflow_list_failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list workflows: {str(e)}"
        )

@router.get("/{workflow_id}", response_model=WorkflowResponse)
async def get_workflow(
    workflow_id: str,
    db: AsyncSession = Depends(get_session_dependency)
) -> WorkflowResponse:
    """
    Get details of a specific workflow.
    
    Performance target: <150ms
    """
    try:
        # Query workflow
        query = select(Workflow).where(Workflow.id == workflow_id)
        result = await db.execute(query)
        workflow = result.scalar_one_or_none()
        
        if not workflow:
            raise HTTPException(
                status_code=404,
                detail=f"Workflow {workflow_id} not found"
            )
            
        return WorkflowResponse.from_orm(workflow)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("workflow_get_failed", workflow_id=workflow_id, error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get workflow: {str(e)}"
        )

@router.put("/{workflow_id}", response_model=WorkflowResponse)
async def update_workflow(
    request: Request,
    workflow_id: str,
    workflow_data: WorkflowUpdate,
    db: AsyncSession = Depends(get_session_dependency),
    workflow_engine: WorkflowEngine = Depends(get_workflow_engine)
) -> WorkflowResponse:
    """
    Update an existing workflow.
    
    Performance target: <150ms
    """
    current_user = get_current_user_from_request(request)
    
    try:
        # Get existing workflow
        query = select(Workflow).where(Workflow.id == workflow_id)
        result = await db.execute(query)
        workflow = result.scalar_one_or_none()
        
        if not workflow:
            raise HTTPException(
                status_code=404,
                detail=f"Workflow {workflow_id} not found"
            )
        
        # Update workflow fields
        update_data = workflow_data.dict(exclude_unset=True)
        
        if update_data:
            update_data["updated_at"] = datetime.utcnow()
            update_data["updated_by"] = current_user.id
            
            # Update in database
            await db.execute(
                update(Workflow)
                .where(Workflow.id == workflow_id)
                .values(**update_data)
            )
            await db.commit()
            
            # Update in workflow engine if definition changed
            if "definition" in update_data or "configuration" in update_data:
                await workflow_engine.update_workflow(
                    workflow_id=workflow_id,
                    definition=update_data.get("definition", workflow.definition),
                    configuration=update_data.get("configuration", workflow.configuration)
                )
        
        # Get updated workflow
        result = await db.execute(query)
        updated_workflow = result.scalar_one()
        
        logger.info(
            "workflow_updated",
            workflow_id=workflow_id,
            updated_by=current_user.id,
            updated_fields=list(update_data.keys())
        )
        
        return WorkflowResponse.from_orm(updated_workflow)
        
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        logger.error("workflow_update_failed", workflow_id=workflow_id, error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update workflow: {str(e)}"
        )

@router.delete("/{workflow_id}", status_code=204)
async def delete_workflow(
    request: Request,
    workflow_id: str,
    db: AsyncSession = Depends(get_session_dependency),
    workflow_engine: WorkflowEngine = Depends(get_workflow_engine)
):
    """
    Delete a workflow from the system.
    
    Performance target: <150ms
    """
    current_user = get_current_user_from_request(request)
    
    try:
        # Check if workflow exists
        query = select(Workflow).where(Workflow.id == workflow_id)
        result = await db.execute(query)
        workflow = result.scalar_one_or_none()
        
        if not workflow:
            raise HTTPException(
                status_code=404,
                detail=f"Workflow {workflow_id} not found"
            )
        
        # Check if workflow is currently running
        if workflow.status == WorkflowStatus.RUNNING:
            raise HTTPException(
                status_code=400,
                detail="Cannot delete a running workflow"
            )
        
        # Unregister from workflow engine
        await workflow_engine.unregister_workflow(workflow_id)
        
        # Delete from database
        await db.execute(delete(Workflow).where(Workflow.id == workflow_id))
        await db.commit()
        
        logger.info(
            "workflow_deleted",
            workflow_id=workflow_id,
            deleted_by=current_user.id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        logger.error("workflow_delete_failed", workflow_id=workflow_id, error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete workflow: {str(e)}"
        )

@router.post("/{workflow_id}/execute")
async def execute_workflow(
    request: Request,
    workflow_id: str,
    execution_data: WorkflowExecutionRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_session_dependency),
    workflow_engine: WorkflowEngine = Depends(get_workflow_engine)
):
    """
    Execute a workflow with given parameters.
    
    Performance target: <150ms (for initiation, actual execution is async)
    """
    current_user = get_current_user_from_request(request)
    
    try:
        # Get workflow
        query = select(Workflow).where(Workflow.id == workflow_id)
        result = await db.execute(query)
        workflow = result.scalar_one_or_none()
        
        if not workflow:
            raise HTTPException(
                status_code=404,
                detail=f"Workflow {workflow_id} not found"
            )
        
        if workflow.status != WorkflowStatus.ACTIVE:
            raise HTTPException(
                status_code=400,
                detail=f"Workflow must be active to execute (current status: {workflow.status.value})"
            )
        
        # Generate execution ID
        execution_id = str(uuid.uuid4())
        
        # Update workflow status to running
        await db.execute(
            update(Workflow)
            .where(Workflow.id == workflow_id)
            .values(
                status=WorkflowStatus.RUNNING,
                updated_at=datetime.utcnow(),
                updated_by=current_user.id
            )
        )
        await db.commit()
        
        # Start workflow execution in background
        background_tasks.add_task(
            _execute_workflow_background,
            workflow_engine,
            workflow_id,
            execution_id,
            execution_data.parameters,
            current_user.id
        )
        
        logger.info(
            "workflow_execution_started",
            workflow_id=workflow_id,
            execution_id=execution_id,
            started_by=current_user.id
        )
        
        return {
            "execution_id": execution_id,
            "workflow_id": workflow_id,
            "status": "started",
            "message": "Workflow execution initiated"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        logger.error("workflow_execution_failed", workflow_id=workflow_id, error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to execute workflow: {str(e)}"
        )

async def _execute_workflow_background(
    workflow_engine: WorkflowEngine,
    workflow_id: str,
    execution_id: str,
    parameters: Dict[str, Any],
    user_id: str
):
    """Background task for workflow execution."""
    try:
        # Execute workflow
        result = await workflow_engine.execute_workflow(
            workflow_id=workflow_id,
            execution_id=execution_id,
            parameters=parameters
        )
        
        # Update workflow status back to active
        async with get_session_dependency() as db:
            await db.execute(
                update(Workflow)
                .where(Workflow.id == workflow_id)
                .values(
                    status=WorkflowStatus.ACTIVE,
                    updated_at=datetime.utcnow()
                )
            )
            await db.commit()
        
        logger.info(
            "workflow_execution_completed",
            workflow_id=workflow_id,
            execution_id=execution_id,
            result=result
        )
        
    except Exception as e:
        # Update workflow status to failed
        async with get_session_dependency() as db:
            await db.execute(
                update(Workflow)
                .where(Workflow.id == workflow_id)
                .values(
                    status=WorkflowStatus.FAILED,
                    updated_at=datetime.utcnow()
                )
            )
            await db.commit()
        
        logger.error(
            "workflow_execution_error",
            workflow_id=workflow_id,
            execution_id=execution_id,
            error=str(e)
        )

@router.get("/{workflow_id}/executions")
async def list_workflow_executions(
    workflow_id: str,
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=1000),
    db: AsyncSession = Depends(get_session_dependency)
):
    """
    List execution history for a workflow.
    
    Performance target: <150ms
    """
    try:
        # Verify workflow exists
        workflow_query = select(Workflow).where(Workflow.id == workflow_id)
        workflow_result = await db.execute(workflow_query)
        workflow = workflow_result.scalar_one_or_none()
        
        if not workflow:
            raise HTTPException(
                status_code=404,
                detail=f"Workflow {workflow_id} not found"
            )
        
        # For now, return placeholder execution history
        # In production, this would query a workflow_executions table
        return {
            "executions": [],
            "total": 0,
            "skip": skip,
            "limit": limit,
            "workflow_id": workflow_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("workflow_executions_list_failed", workflow_id=workflow_id, error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list workflow executions: {str(e)}"
        )

@router.get("/{workflow_id}/stats", response_model=WorkflowStatsResponse)
async def get_workflow_stats(
    workflow_id: str,
    db: AsyncSession = Depends(get_session_dependency)
) -> WorkflowStatsResponse:
    """
    Get performance statistics for a workflow.
    
    Performance target: <150ms
    """
    try:
        # Verify workflow exists
        workflow_query = select(Workflow).where(Workflow.id == workflow_id)
        workflow_result = await db.execute(workflow_query)
        workflow = workflow_result.scalar_one_or_none()
        
        if not workflow:
            raise HTTPException(
                status_code=404,
                detail=f"Workflow {workflow_id} not found"
            )
        
        # Calculate statistics (placeholder for now)
        # In production, this would aggregate from workflow_executions
        return WorkflowStatsResponse(
            workflow_id=workflow_id,
            total_executions=0,
            successful_executions=0,
            failed_executions=0,
            average_duration_seconds=0,
            success_rate=0,
            last_execution=None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("workflow_stats_failed", workflow_id=workflow_id, error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get workflow stats: {str(e)}"
        )