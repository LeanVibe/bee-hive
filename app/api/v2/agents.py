"""
Epic B: Core Agent Management API Endpoints

Provides working agent creation, monitoring, and management functionality
required by the Mobile PWA. Implements basic agent lifecycle operations
following the first principles approach from the strategic plan.
"""

import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel, Field
import structlog

from ...core.simple_orchestrator import SimpleOrchestrator, AgentRole, AgentInstance, create_simple_orchestrator
from ...core.database import get_session_dependency
from ...models.agent import AgentStatus, AgentType

logger = structlog.get_logger()
router = APIRouter(prefix="/agents", tags=["agents"])

# Simple orchestrator instance for Epic B
_orchestrator: Optional[SimpleOrchestrator] = None

async def get_orchestrator() -> SimpleOrchestrator:
    """Get orchestrator instance with WebSocket integration, creating if needed."""
    global _orchestrator
    if _orchestrator is None:
        # Import WebSocket manager
        from .websockets import manager as websocket_manager
        
        _orchestrator = create_simple_orchestrator(websocket_manager=websocket_manager)
        await _orchestrator.initialize()
    return _orchestrator


# Pydantic models for API
class AgentCreateRequest(BaseModel):
    """Request model for agent creation."""
    role: str = Field(..., description="Agent role (backend_developer, frontend_developer, etc.)")
    agent_type: str = Field(default="claude_code", description="Type of agent to create")
    task_id: Optional[str] = Field(None, description="Optional task ID to assign")
    workspace_name: Optional[str] = Field(None, description="Workspace name")
    git_branch: Optional[str] = Field(None, description="Git branch for workspace")


class AgentResponse(BaseModel):
    """Response model for agent data."""
    id: str
    role: str
    status: str
    created_at: str
    last_activity: str
    current_task_id: Optional[str] = None


class AgentListResponse(BaseModel):
    """Response model for agent list."""
    agents: List[AgentResponse]
    total: int
    active: int
    inactive: int


class AgentStatusUpdate(BaseModel):
    """Request model for agent status updates."""
    status: str = Field(..., description="New agent status")


@router.post("/", response_model=AgentResponse, status_code=201)
async def create_agent(
    request: AgentCreateRequest,
    background_tasks: BackgroundTasks
) -> AgentResponse:
    """
    Create a new agent instance.
    
    Epic B Phase B.1: Basic agent creation functionality
    Required by Mobile PWA for agent management.
    """
    try:
        orchestrator = await get_orchestrator()
        
        # Map string role to enum
        try:
            agent_role = AgentRole(request.role.lower())
        except ValueError:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid role. Valid roles: {[r.value for r in AgentRole]}"
            )
        
        # Spawn agent using simple orchestrator
        agent_id = await orchestrator.spawn_agent(
            role=agent_role,
            task_id=request.task_id,
            workspace_name=request.workspace_name,
            git_branch=request.git_branch
        )
        
        # Get agent details
        agent = orchestrator._agents.get(agent_id)
        if not agent:
            raise HTTPException(status_code=500, detail="Agent created but not found in registry")
        
        logger.info("Agent created via API", agent_id=agent_id, role=request.role)
        
        return AgentResponse(
            id=agent_id,
            role=agent.role.value,
            status=agent.status.value,
            created_at=agent.created_at.isoformat(),
            last_activity=agent.last_activity.isoformat(),
            current_task_id=agent.current_task_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to create agent via API", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to create agent")


@router.get("/", response_model=AgentListResponse)
async def list_agents(
    status: Optional[str] = Query(None, description="Filter by status"),
    role: Optional[str] = Query(None, description="Filter by role"),
    limit: int = Query(50, ge=1, le=100, description="Max agents to return"),
    offset: int = Query(0, ge=0, description="Agents to skip")
) -> AgentListResponse:
    """
    List all agents with filtering.
    
    Epic B Phase B.1: Agent listing required by Mobile PWA dashboard.
    """
    try:
        orchestrator = await get_orchestrator()
        agents = list(orchestrator._agents.values())
        
        # Apply filters
        if status:
            agents = [a for a in agents if a.status.value == status]
        if role:
            agents = [a for a in agents if a.role.value == role]
        
        # Apply pagination
        total = len(agents)
        agents = agents[offset:offset + limit]
        
        # Convert to response format
        agent_responses = []
        for agent in agents:
            agent_responses.append(AgentResponse(
                id=agent.id,
                role=agent.role.value,
                status=agent.status.value,
                created_at=agent.created_at.isoformat(),
                last_activity=agent.last_activity.isoformat(),
                current_task_id=agent.current_task_id
            ))
        
        # Calculate stats
        active_count = len([a for a in orchestrator._agents.values() 
                           if a.status == AgentStatus.ACTIVE])
        inactive_count = len([a for a in orchestrator._agents.values() 
                             if a.status == AgentStatus.INACTIVE])
        
        return AgentListResponse(
            agents=agent_responses,
            total=total,
            active=active_count,
            inactive=inactive_count
        )
        
    except Exception as e:
        logger.error("Failed to list agents", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to list agents")


@router.get("/{agent_id}", response_model=AgentResponse)
async def get_agent(agent_id: str) -> AgentResponse:
    """
    Get specific agent details.
    
    Epic B Phase B.2: Agent monitoring and status tracking.
    """
    try:
        orchestrator = await get_orchestrator()
        agent = orchestrator._agents.get(agent_id)
        
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        return AgentResponse(
            id=agent.id,
            role=agent.role.value,
            status=agent.status.value,
            created_at=agent.created_at.isoformat(),
            last_activity=agent.last_activity.isoformat(),
            current_task_id=agent.current_task_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get agent", agent_id=agent_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to get agent")


@router.put("/{agent_id}/status", response_model=AgentResponse)
async def update_agent_status(
    agent_id: str,
    request: AgentStatusUpdate
) -> AgentResponse:
    """
    Update agent status.
    
    Epic B Phase B.2: Agent control and status management.
    """
    try:
        orchestrator = await get_orchestrator()
        agent = orchestrator._agents.get(agent_id)
        
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        # Validate status
        try:
            new_status = AgentStatus(request.status.upper())
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid status. Valid statuses: {[s.value for s in AgentStatus]}"
            )
        
        # Update agent status
        old_status = agent.status
        agent.status = new_status
        agent.last_activity = datetime.utcnow()
        
        # Update in database if available
        await orchestrator._update_agent_status(agent_id, new_status)
        
        logger.info("Agent status updated", 
                   agent_id=agent_id, 
                   old_status=old_status.value, 
                   new_status=new_status.value)
        
        return AgentResponse(
            id=agent.id,
            role=agent.role.value,
            status=agent.status.value,
            created_at=agent.created_at.isoformat(),
            last_activity=agent.last_activity.isoformat(),
            current_task_id=agent.current_task_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to update agent status", agent_id=agent_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to update agent status")


@router.delete("/{agent_id}", status_code=204)
async def delete_agent(agent_id: str):
    """
    Terminate and delete an agent.
    
    Epic B Phase B.1: Agent termination functionality.
    """
    try:
        orchestrator = await get_orchestrator()
        
        success = await orchestrator.shutdown_agent(agent_id, graceful=True)
        if not success:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        logger.info("Agent terminated via API", agent_id=agent_id)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to delete agent", agent_id=agent_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to delete agent")


@router.get("/{agent_id}/health")
async def get_agent_health(agent_id: str) -> Dict[str, Any]:
    """
    Get agent health status.
    
    Epic B Phase B.2: Health monitoring for agents.
    """
    try:
        orchestrator = await get_orchestrator()
        agent = orchestrator._agents.get(agent_id)
        
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        # Basic health metrics
        health_data = {
            "agent_id": agent_id,
            "status": agent.status.value,
            "last_activity": agent.last_activity.isoformat(),
            "uptime_seconds": (datetime.utcnow() - agent.created_at).total_seconds(),
            "current_task": agent.current_task_id,
            "healthy": agent.status in [AgentStatus.ACTIVE, AgentStatus.IDLE]
        }
        
        # Add tmux session info if available
        if agent_id in orchestrator._tmux_agents:
            session_info = await orchestrator.get_agent_session_info(agent_id)
            if session_info:
                health_data["session"] = session_info
        
        return health_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get agent health", agent_id=agent_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to get agent health")