"""
Simple Agent API endpoints for LeanVibe Agent Hive 2.0 - Epic 1 Phase 1.1

Provides basic agent CRUD operations that connect to SimpleOrchestrator.
This is the API-Orchestrator integration layer that enables CLI functionality.
"""

import uuid
from typing import Dict, Any, List, Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import BaseModel, Field
import structlog

logger = structlog.get_logger()

# Create router
router = APIRouter()

# Simple request/response models
class AgentCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100, description="Agent name")
    type: str = Field(default="general", description="Agent type: backend_developer, frontend_developer, devops_engineer, qa_engineer, or general")
    capabilities: List[str] = Field(default_factory=list, description="Agent capabilities")

class AgentResponse(BaseModel):
    id: str
    name: str
    type: str
    status: str
    created_at: str
    capabilities: List[str] = []

class AgentListResponse(BaseModel):
    agents: List[AgentResponse]
    total: int


async def get_simple_orchestrator():
    """Get SimpleOrchestrator instance, ensuring it's initialized."""
    try:
        from ...core.simple_orchestrator import get_simple_orchestrator as get_singleton
        orchestrator = get_singleton()
        
        # Ensure the orchestrator is initialized
        if not hasattr(orchestrator, '_initialized') or not orchestrator._initialized:
            await orchestrator.initialize()
            
        return orchestrator
    except Exception as e:
        logger.error("Failed to get SimpleOrchestrator", error=str(e))
        raise HTTPException(
            status_code=503, 
            detail="Agent orchestrator not available"
        )


@router.post("/", response_model=AgentResponse, status_code=status.HTTP_201_CREATED)
async def create_agent(
    request: AgentCreateRequest,
    orchestrator = Depends(get_simple_orchestrator)
) -> AgentResponse:
    """Create a new agent via SimpleOrchestrator."""
    try:
        from ...core.simple_orchestrator import AgentRole, AgentLauncherType
        
        # Map request type to AgentRole
        role_mapping = {
            "backend_developer": AgentRole.BACKEND_DEVELOPER,
            "frontend_developer": AgentRole.FRONTEND_DEVELOPER,
            "devops_engineer": AgentRole.DEVOPS_ENGINEER,
            "qa_engineer": AgentRole.QA_ENGINEER,
            "general": AgentRole.BACKEND_DEVELOPER  # Default fallback
        }
        
        role = role_mapping.get(request.type, AgentRole.BACKEND_DEVELOPER)
        
        # Create agent using SimpleOrchestrator
        agent_id = await orchestrator.spawn_agent(
            role=role,
            agent_id=request.name,  # Use name as agent_id
            agent_type=AgentLauncherType.CLAUDE_CODE,
            environment_vars={"LEANVIBE_AGENT_NAME": request.name}
        )
        
        # Return response
        response = AgentResponse(
            id=str(agent_id),
            name=request.name,
            type=request.type,
            status="created",
            created_at=datetime.utcnow().isoformat(),
            capabilities=request.capabilities
        )
        
        logger.info(
            "Agent created via API", 
            agent_id=str(agent_id),
            name=request.name,
            type=request.type
        )
        
        return response
        
    except Exception as e:
        logger.error("Failed to create agent", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Agent creation failed: {str(e)}"
        )


@router.get("/", response_model=AgentListResponse)
async def list_agents(
    orchestrator = Depends(get_simple_orchestrator)
) -> AgentListResponse:
    """List all agents from SimpleOrchestrator."""
    try:
        
        # Get agents from SimpleOrchestrator
        agents_data = await orchestrator.list_agent_sessions()
        
        # Convert to response format
        agents = []
        if isinstance(agents_data, list):
            # Handle list of agent sessions from SimpleOrchestrator
            for agent_session in agents_data:
                agents.append(AgentResponse(
                    id=str(agent_session.get("agent_id", "unknown")),
                    name=agent_session.get("session_info", {}).get("environment_vars", {}).get("LEANVIBE_AGENT_NAME", agent_session.get("agent_id", "Unknown")),
                    type=agent_session.get("session_info", {}).get("environment_vars", {}).get("LEANVIBE_AGENT_TYPE", "general"),
                    status="active" if agent_session.get("is_running", False) else "inactive",
                    created_at=agent_session.get("session_info", {}).get("created_at", datetime.utcnow().isoformat()),
                    capabilities=[]
                ))
        
        response = AgentListResponse(
            agents=agents,
            total=len(agents)
        )
        
        logger.info("Listed agents via API", count=len(agents))
        return response
        
    except Exception as e:
        logger.error("Failed to list agents", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve agents: {str(e)}"
        )


@router.get("/{agent_id}", response_model=AgentResponse)
async def get_agent(
    agent_id: str,
    orchestrator = Depends(get_simple_orchestrator)
) -> AgentResponse:
    """Get a specific agent by ID."""
    try:
        
        # Get agent from SimpleOrchestrator
        agent_data = await orchestrator.get_agent_session_info(agent_id)
        
        if not agent_data:
            raise HTTPException(
                status_code=404,
                detail="Agent not found"
            )
        
        # Convert to response format
        response = AgentResponse(
            id=str(agent_data.get("agent_id", agent_id)),
            name=agent_data.get("session_info", {}).get("environment_vars", {}).get("LEANVIBE_AGENT_NAME", agent_data.get("agent_id", "Unknown")),
            type=agent_data.get("session_info", {}).get("environment_vars", {}).get("LEANVIBE_AGENT_TYPE", "general"),
            status="active" if agent_data.get("is_running", False) else "inactive",
            created_at=agent_data.get("session_info", {}).get("created_at", datetime.utcnow().isoformat()),
            capabilities=[]
        )
        
        logger.info("Retrieved agent via API", agent_id=agent_id)
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get agent", agent_id=agent_id, error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve agent: {str(e)}"
        )


@router.delete("/{agent_id}", status_code=status.HTTP_200_OK)
async def shutdown_agent(
    agent_id: str,
    graceful: bool = True,
    orchestrator = Depends(get_simple_orchestrator)
) -> Dict[str, Any]:
    """Shutdown an agent via SimpleOrchestrator."""
    try:
        
        # Shutdown agent
        success = await orchestrator.shutdown_agent(agent_id, graceful=graceful)
        
        if not success:
            raise HTTPException(
                status_code=400,
                detail="Failed to shutdown agent"
            )
        
        logger.info("Agent shutdown via API", agent_id=agent_id, graceful=graceful)
        
        return {
            "success": True,
            "agent_id": agent_id,
            "message": "Agent shutdown successfully",
            "graceful": graceful
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to shutdown agent", agent_id=agent_id, error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to shutdown agent: {str(e)}"
        )


@router.get("/{agent_id}/status")
async def get_agent_status(
    agent_id: str,
    orchestrator = Depends(get_simple_orchestrator)
) -> Dict[str, Any]:
    """Get agent status from SimpleOrchestrator."""
    try:
        
        # Get agent status
        agent_data = await orchestrator.get_agent_session_info(agent_id)
        
        if not agent_data:
            raise HTTPException(
                status_code=404,
                detail="Agent not found"
            )
        
        # Return status information
        return {
            "agent_id": agent_id,
            "status": "active" if agent_data.get("is_running", False) else "inactive",
            "name": agent_data.get("session_info", {}).get("environment_vars", {}).get("LEANVIBE_AGENT_NAME", agent_data.get("agent_id", "Unknown")),
            "type": agent_data.get("session_info", {}).get("environment_vars", {}).get("LEANVIBE_AGENT_TYPE", "general"),
            "active_tasks": agent_data.get("metrics", {}).get("task_count", 0),
            "last_active": agent_data.get("session_info", {}).get("last_activity", datetime.utcnow().isoformat()),
            "is_running": agent_data.get("is_running", False),
            "session_info": agent_data.get("session_info", {}),
            "metrics": agent_data.get("metrics", {})
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get agent status", agent_id=agent_id, error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get agent status: {str(e)}"
        )


# System status endpoint
@router.get("/system/status")
async def get_system_status(
    orchestrator = Depends(get_simple_orchestrator)
) -> Dict[str, Any]:
    """Get system status from SimpleOrchestrator."""
    try:
        # Get system status
        status_data = await orchestrator.get_system_status()
        
        return {
            "success": True,
            "system_status": status_data,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error("Failed to get system status", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get system status: {str(e)}"
        )