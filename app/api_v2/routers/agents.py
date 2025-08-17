"""
Agents API - Consolidated agent management endpoints

Consolidates agent_activation.py, agent_coordination.py, v1/agents.py,
v1/autonomous_development.py, and v1/autonomous_self_modification.py
into a unified RESTful resource for agent CRUD and lifecycle operations.

Performance target: <100ms P95 response time
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
from ...core.orchestrator import AgentOrchestrator
from ...models.agent import Agent, AgentStatus, AgentType
from ...models.task import Task, TaskStatus
from ...schemas.agent import (
    AgentCreateRequest,
    AgentUpdateRequest, 
    AgentResponse,
    AgentListResponse,
    AgentStatsResponse,
    AgentActivationRequest
)
from ..middleware import (
    require_permission,
    require_role,
    get_current_user_from_request
)
from ...core.auth import Permission, UserRole

logger = structlog.get_logger()
router = APIRouter()

# Agent orchestrator dependency
async def get_agent_orchestrator() -> AgentOrchestrator:
    """Get agent orchestrator instance."""
    # In production, this would be dependency-injected
    return AgentOrchestrator()

@router.post("/", response_model=AgentResponse, status_code=201)
async def create_agent(
    request: Request,
    agent_data: AgentCreateRequest,
    db: AsyncSession = Depends(get_session_dependency),
    orchestrator: AgentOrchestrator = Depends(get_agent_orchestrator)
) -> AgentResponse:
    """
    Create a new AI agent in the multi-agent system.
    
    Requires CREATE_DEVELOPMENT_TASK permission.
    Performance target: <100ms
    """
    # Permission check handled by middleware, but add explicit check for clarity
    current_user = get_current_user_from_request(request)
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    try:
        # Create agent record
        agent = Agent(
            id=str(uuid.uuid4()),
            name=agent_data.name,
            type=agent_data.type,
            status=AgentStatus.CREATED,
            capabilities=agent_data.capabilities,
            configuration=agent_data.configuration or {},
            metadata={
                "created_by": current_user.id,
                "created_at": datetime.utcnow().isoformat(),
                "version": "2.0"
            }
        )
        
        db.add(agent)
        await db.commit()
        await db.refresh(agent)
        
        # Initialize agent in orchestrator
        await orchestrator.register_agent(
            agent_id=agent.id,
            agent_type=agent.type,
            capabilities=agent.capabilities
        )
        
        logger.info(
            "agent_created",
            agent_id=agent.id,
            agent_name=agent.name,
            agent_type=agent.type.value,
            created_by=current_user.id
        )
        
        return AgentResponse.from_orm(agent)
        
    except Exception as e:
        await db.rollback()
        logger.error("agent_creation_failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create agent: {str(e)}"
        )

@router.get("/", response_model=AgentListResponse)
async def list_agents(
    request: Request,
    skip: int = Query(0, ge=0, description="Number of agents to skip"),
    limit: int = Query(50, ge=1, le=1000, description="Number of agents to return"),
    status: Optional[AgentStatus] = Query(None, description="Filter by agent status"),
    type: Optional[AgentType] = Query(None, description="Filter by agent type"),
    db: AsyncSession = Depends(get_session_dependency)
) -> AgentListResponse:
    """
    List all agents with optional filtering.
    
    Performance target: <100ms
    """
    try:
        # Build query with filters
        query = select(Agent)
        
        if status:
            query = query.where(Agent.status == status)
        if type:
            query = query.where(Agent.type == type)
            
        # Apply pagination
        query = query.offset(skip).limit(limit)
        
        # Execute query
        result = await db.execute(query)
        agents = result.scalars().all()
        
        # Get total count for pagination
        count_query = select(Agent)
        if status:
            count_query = count_query.where(Agent.status == status)
        if type:
            count_query = count_query.where(Agent.type == type)
            
        total_result = await db.execute(count_query)
        total = len(total_result.scalars().all())
        
        return AgentListResponse(
            agents=[AgentResponse.from_orm(agent) for agent in agents],
            total=total,
            skip=skip,
            limit=limit
        )
        
    except Exception as e:
        logger.error("agent_list_failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list agents: {str(e)}"
        )

@router.get("/{agent_id}", response_model=AgentResponse)
async def get_agent(
    agent_id: str,
    db: AsyncSession = Depends(get_session_dependency)
) -> AgentResponse:
    """
    Get details of a specific agent.
    
    Performance target: <100ms
    """
    try:
        # Query agent with related data
        query = select(Agent).where(Agent.id == agent_id)
        result = await db.execute(query)
        agent = result.scalar_one_or_none()
        
        if not agent:
            raise HTTPException(
                status_code=404,
                detail=f"Agent {agent_id} not found"
            )
            
        return AgentResponse.from_orm(agent)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("agent_get_failed", agent_id=agent_id, error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get agent: {str(e)}"
        )

@router.put("/{agent_id}", response_model=AgentResponse)
async def update_agent(
    request: Request,
    agent_id: str,
    agent_data: AgentUpdateRequest,
    db: AsyncSession = Depends(get_session_dependency),
    orchestrator: AgentOrchestrator = Depends(get_agent_orchestrator)
) -> AgentResponse:
    """
    Update an existing agent.
    
    Requires UPDATE_PILOT permission for configuration changes.
    Performance target: <100ms
    """
    current_user = get_current_user_from_request(request)
    
    try:
        # Get existing agent
        query = select(Agent).where(Agent.id == agent_id)
        result = await db.execute(query)
        agent = result.scalar_one_or_none()
        
        if not agent:
            raise HTTPException(
                status_code=404,
                detail=f"Agent {agent_id} not found"
            )
        
        # Update agent fields
        update_data = agent_data.dict(exclude_unset=True)
        
        if update_data:
            update_data["updated_at"] = datetime.utcnow()
            update_data["updated_by"] = current_user.id
            
            # Update in database
            await db.execute(
                update(Agent)
                .where(Agent.id == agent_id)
                .values(**update_data)
            )
            await db.commit()
            
            # Update in orchestrator if capabilities changed
            if "capabilities" in update_data:
                await orchestrator.update_agent_capabilities(
                    agent_id=agent_id,
                    capabilities=update_data["capabilities"]
                )
        
        # Get updated agent
        result = await db.execute(query)
        updated_agent = result.scalar_one()
        
        logger.info(
            "agent_updated",
            agent_id=agent_id,
            updated_by=current_user.id,
            updated_fields=list(update_data.keys())
        )
        
        return AgentResponse.from_orm(updated_agent)
        
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        logger.error("agent_update_failed", agent_id=agent_id, error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update agent: {str(e)}"
        )

@router.delete("/{agent_id}", status_code=204)
async def delete_agent(
    request: Request,
    agent_id: str,
    db: AsyncSession = Depends(get_session_dependency),
    orchestrator: AgentOrchestrator = Depends(get_agent_orchestrator)
):
    """
    Delete an agent from the system.
    
    Requires DELETE_PILOT permission.
    Performance target: <100ms
    """
    current_user = get_current_user_from_request(request)
    
    try:
        # Check if agent exists
        query = select(Agent).where(Agent.id == agent_id)
        result = await db.execute(query)
        agent = result.scalar_one_or_none()
        
        if not agent:
            raise HTTPException(
                status_code=404,
                detail=f"Agent {agent_id} not found"
            )
        
        # Check if agent has active tasks
        active_tasks_query = select(Task).where(
            and_(
                Task.assigned_agent_id == agent_id,
                Task.status.in_([TaskStatus.PENDING, TaskStatus.RUNNING])
            )
        )
        active_tasks_result = await db.execute(active_tasks_query)
        active_tasks = active_tasks_result.scalars().all()
        
        if active_tasks:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot delete agent with {len(active_tasks)} active tasks"
            )
        
        # Deregister from orchestrator
        await orchestrator.deregister_agent(agent_id)
        
        # Delete from database
        await db.execute(delete(Agent).where(Agent.id == agent_id))
        await db.commit()
        
        logger.info(
            "agent_deleted",
            agent_id=agent_id,
            deleted_by=current_user.id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        logger.error("agent_delete_failed", agent_id=agent_id, error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete agent: {str(e)}"
        )

@router.post("/{agent_id}/activate", response_model=AgentResponse)
async def activate_agent(
    request: Request,
    agent_id: str,
    activation_data: AgentActivationRequest,
    db: AsyncSession = Depends(get_session_dependency),
    orchestrator: AgentOrchestrator = Depends(get_agent_orchestrator)
) -> AgentResponse:
    """
    Activate an agent for task execution.
    
    Performance target: <100ms
    """
    current_user = get_current_user_from_request(request)
    
    try:
        # Get agent
        query = select(Agent).where(Agent.id == agent_id)
        result = await db.execute(query)
        agent = result.scalar_one_or_none()
        
        if not agent:
            raise HTTPException(
                status_code=404,
                detail=f"Agent {agent_id} not found"
            )
        
        # Activate in orchestrator
        await orchestrator.activate_agent(
            agent_id=agent_id,
            configuration=activation_data.configuration
        )
        
        # Update status in database
        await db.execute(
            update(Agent)
            .where(Agent.id == agent_id)
            .values(
                status=AgentStatus.ACTIVE,
                updated_at=datetime.utcnow(),
                updated_by=current_user.id
            )
        )
        await db.commit()
        
        # Get updated agent
        result = await db.execute(query)
        activated_agent = result.scalar_one()
        
        logger.info(
            "agent_activated",
            agent_id=agent_id,
            activated_by=current_user.id
        )
        
        return AgentResponse.from_orm(activated_agent)
        
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        logger.error("agent_activation_failed", agent_id=agent_id, error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to activate agent: {str(e)}"
        )

@router.post("/{agent_id}/deactivate", response_model=AgentResponse)
async def deactivate_agent(
    request: Request,
    agent_id: str,
    db: AsyncSession = Depends(get_session_dependency),
    orchestrator: AgentOrchestrator = Depends(get_agent_orchestrator)
) -> AgentResponse:
    """
    Deactivate an agent from task execution.
    
    Performance target: <100ms
    """
    current_user = get_current_user_from_request(request)
    
    try:
        # Get agent
        query = select(Agent).where(Agent.id == agent_id)
        result = await db.execute(query)
        agent = result.scalar_one_or_none()
        
        if not agent:
            raise HTTPException(
                status_code=404,
                detail=f"Agent {agent_id} not found"
            )
        
        # Deactivate in orchestrator
        await orchestrator.deactivate_agent(agent_id)
        
        # Update status in database
        await db.execute(
            update(Agent)
            .where(Agent.id == agent_id)
            .values(
                status=AgentStatus.INACTIVE,
                updated_at=datetime.utcnow(),
                updated_by=current_user.id
            )
        )
        await db.commit()
        
        # Get updated agent
        result = await db.execute(query)
        deactivated_agent = result.scalar_one()
        
        logger.info(
            "agent_deactivated",
            agent_id=agent_id,
            deactivated_by=current_user.id
        )
        
        return AgentResponse.from_orm(deactivated_agent)
        
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        logger.error("agent_deactivation_failed", agent_id=agent_id, error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to deactivate agent: {str(e)}"
        )

@router.get("/{agent_id}/tasks")
async def list_agent_tasks(
    agent_id: str,
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=1000),
    status: Optional[TaskStatus] = Query(None),
    db: AsyncSession = Depends(get_session_dependency)
):
    """
    List tasks assigned to a specific agent.
    
    Performance target: <100ms
    """
    try:
        # Verify agent exists
        agent_query = select(Agent).where(Agent.id == agent_id)
        agent_result = await db.execute(agent_query)
        agent = agent_result.scalar_one_or_none()
        
        if not agent:
            raise HTTPException(
                status_code=404,
                detail=f"Agent {agent_id} not found"
            )
        
        # Query tasks
        query = select(Task).where(Task.assigned_agent_id == agent_id)
        
        if status:
            query = query.where(Task.status == status)
            
        query = query.offset(skip).limit(limit)
        
        result = await db.execute(query)
        tasks = result.scalars().all()
        
        # Get total count
        count_query = select(Task).where(Task.assigned_agent_id == agent_id)
        if status:
            count_query = count_query.where(Task.status == status)
            
        total_result = await db.execute(count_query)
        total = len(total_result.scalars().all())
        
        return {
            "tasks": [
                {
                    "id": task.id,
                    "title": task.title,
                    "status": task.status.value,
                    "priority": task.priority.value,
                    "created_at": task.created_at.isoformat(),
                    "updated_at": task.updated_at.isoformat() if task.updated_at else None
                }
                for task in tasks
            ],
            "total": total,
            "skip": skip,
            "limit": limit,
            "agent_id": agent_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("agent_tasks_list_failed", agent_id=agent_id, error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list agent tasks: {str(e)}"
        )

@router.get("/{agent_id}/stats", response_model=AgentStatsResponse)
async def get_agent_stats(
    agent_id: str,
    db: AsyncSession = Depends(get_session_dependency)
) -> AgentStatsResponse:
    """
    Get performance statistics for a specific agent.
    
    Performance target: <100ms
    """
    try:
        # Verify agent exists
        agent_query = select(Agent).where(Agent.id == agent_id)
        agent_result = await db.execute(agent_query)
        agent = agent_result.scalar_one_or_none()
        
        if not agent:
            raise HTTPException(
                status_code=404,
                detail=f"Agent {agent_id} not found"
            )
        
        # Get task statistics
        tasks_query = select(Task).where(Task.assigned_agent_id == agent_id)
        tasks_result = await db.execute(tasks_query)
        tasks = tasks_result.scalars().all()
        
        # Calculate statistics
        total_tasks = len(tasks)
        completed_tasks = len([t for t in tasks if t.status == TaskStatus.COMPLETED])
        failed_tasks = len([t for t in tasks if t.status == TaskStatus.FAILED])
        active_tasks = len([t for t in tasks if t.status in [TaskStatus.PENDING, TaskStatus.RUNNING]])
        
        success_rate = (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0
        
        return AgentStatsResponse(
            agent_id=agent_id,
            total_tasks=total_tasks,
            completed_tasks=completed_tasks,
            failed_tasks=failed_tasks,
            active_tasks=active_tasks,
            success_rate=round(success_rate, 2),
            uptime_hours=0,  # Would be calculated from agent lifecycle data
            last_activity=agent.updated_at or agent.created_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("agent_stats_failed", agent_id=agent_id, error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get agent stats: {str(e)}"
        )