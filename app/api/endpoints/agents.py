"""
Agent Management API Endpoints - Epic C Phase 1

Core API endpoints for agent lifecycle management, status control, and orchestrator integration.
Follows established FastAPI patterns from business_analytics.py for consistency and reliability.

Key Features:
- Agent creation and deletion with proper validation
- Status control (activate, deactivate, monitor)
- Integration with orchestrator layer for real operations
- Response times <200ms per Epic requirements
- Comprehensive error handling and logging

Epic C Phase 1: API Endpoint Implementation
"""

import asyncio
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any

from fastapi import APIRouter, HTTPException, Depends, Query, Path, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ...schemas.agent import (
    AgentCreate, AgentUpdate, AgentResponse, AgentListResponse,
    AgentStatsResponse, AgentActivationRequest
)
from ...models.agent import AgentStatus, AgentType
from ...core.database import get_async_session
from ...core.logging_service import get_component_logger
from ...core.simple_orchestrator import SimpleOrchestrator

# Initialize logging
logger = get_component_logger("agents_api")

# Create router
router = APIRouter(prefix="/api/v1/agents", tags=["agents"])


class AgentOperationResponse(BaseModel):
    """Standard response model for agent operations."""
    success: bool = True
    message: str
    agent_id: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class AgentStatusUpdateRequest(BaseModel):
    """Request model for agent status updates."""
    status: AgentStatus = Field(..., description="New agent status")
    reason: Optional[str] = Field(None, description="Reason for status change")


async def get_orchestrator() -> SimpleOrchestrator:
    """Dependency to get orchestrator instance."""
    try:
        # Try to get or create orchestrator instance
        # TODO: Replace with proper dependency injection
        orchestrator = SimpleOrchestrator()
        await orchestrator.initialize()
        return orchestrator
    except Exception as e:
        logger.warning(f"Orchestrator unavailable: {e}")
        raise HTTPException(
            status_code=503,
            detail="Orchestration service temporarily unavailable"
        )


@router.post("/", response_model=AgentResponse, status_code=201)
async def create_agent(
    agent_data: AgentCreate,
    background_tasks: BackgroundTasks,
    orchestrator: SimpleOrchestrator = Depends(get_orchestrator)
):
    """
    Create a new agent in the system.
    
    Creates an agent with the specified configuration and registers it with
    the orchestrator for task assignment and lifecycle management.
    
    **Performance Target**: <200ms response time
    **Integration**: Full orchestrator integration for production usage
    """
    try:
        start_time = datetime.utcnow()
        
        # Generate unique agent ID
        agent_id = str(uuid.uuid4())
        
        # Create agent configuration
        agent_config = {
            "id": agent_id,
            "name": agent_data.name,
            "type": agent_data.type,
            "role": agent_data.role,
            "capabilities": agent_data.capabilities or [],
            "system_prompt": agent_data.system_prompt,
            "config": agent_data.config or {},
            "status": AgentStatus.CREATED,
            "created_at": start_time.isoformat(),
            "updated_at": start_time.isoformat()
        }
        
        # Register agent with orchestrator
        try:
            orchestrator_result = await orchestrator.create_agent(agent_config)
            logger.info(f"Agent {agent_data.name} registered with orchestrator: {orchestrator_result}")
        except Exception as e:
            logger.warning(f"Orchestrator registration failed for agent {agent_data.name}: {e}")
            # Continue with creation but mark status appropriately
            agent_config["status"] = AgentStatus.INACTIVE
        
        # Prepare response data
        response_data = AgentResponse(
            id=uuid.UUID(agent_id),
            name=agent_data.name,
            type=agent_data.type,
            role=agent_data.role,
            capabilities=agent_data.capabilities or [],
            status=agent_config["status"],
            config=agent_data.config or {},
            tmux_session=None,  # Will be populated when agent starts
            total_tasks_completed=0,
            total_tasks_failed=0,
            average_response_time=0.0,
            context_window_usage=0.0,
            created_at=start_time,
            updated_at=start_time,
            last_heartbeat=None,
            last_active=None
        )
        
        # Performance monitoring
        elapsed_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        if elapsed_ms > 200:
            logger.warning(f"Agent creation took {elapsed_ms:.1f}ms (target: <200ms)")
        
        logger.info(f"✅ Created agent: {agent_data.name} ({agent_id}) in {elapsed_ms:.1f}ms")
        return response_data
        
    except Exception as e:
        logger.error(f"Failed to create agent {agent_data.name}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Agent creation failed: {str(e)}"
        )


@router.get("/", response_model=AgentListResponse)
async def list_agents(
    status: Optional[AgentStatus] = Query(None, description="Filter by agent status"),
    agent_type: Optional[AgentType] = Query(None, description="Filter by agent type"),
    limit: int = Query(20, ge=1, le=100, description="Maximum number of agents to return"),
    offset: int = Query(0, ge=0, description="Number of agents to skip"),
    orchestrator: SimpleOrchestrator = Depends(get_orchestrator)
):
    """
    List all agents with optional filtering.
    
    Retrieves agents from the orchestrator with current status and performance metrics.
    Supports pagination and filtering for efficient querying.
    """
    try:
        start_time = datetime.utcnow()
        
        # Get agents from orchestrator
        try:
            agents_data = await orchestrator.list_agents(
                status_filter=status,
                type_filter=agent_type,
                limit=limit,
                offset=offset
            )
        except Exception as e:
            logger.warning(f"Orchestrator query failed: {e}")
            # Return empty list if orchestrator unavailable
            agents_data = {"agents": [], "total": 0}
        
        # Convert to response format
        agents = []
        for agent_data in agents_data.get("agents", []):
            try:
                agent_response = AgentResponse(
                    id=uuid.UUID(agent_data["id"]),
                    name=agent_data["name"],
                    type=agent_data.get("type", AgentType.CLAUDE),
                    role=agent_data.get("role"),
                    capabilities=agent_data.get("capabilities", []),
                    status=agent_data.get("status", AgentStatus.UNKNOWN),
                    config=agent_data.get("config", {}),
                    tmux_session=agent_data.get("tmux_session"),
                    total_tasks_completed=agent_data.get("total_tasks_completed", 0),
                    total_tasks_failed=agent_data.get("total_tasks_failed", 0),
                    average_response_time=agent_data.get("average_response_time", 0.0),
                    context_window_usage=agent_data.get("context_window_usage", 0.0),
                    created_at=agent_data.get("created_at"),
                    updated_at=agent_data.get("updated_at"),
                    last_heartbeat=agent_data.get("last_heartbeat"),
                    last_active=agent_data.get("last_active")
                )
                agents.append(agent_response)
            except Exception as e:
                logger.warning(f"Skipping malformed agent data: {e}")
                continue
        
        response = AgentListResponse(
            agents=agents,
            total=agents_data.get("total", len(agents)),
            offset=offset,
            limit=limit
        )
        
        # Performance monitoring
        elapsed_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        logger.info(f"Listed {len(agents)} agents in {elapsed_ms:.1f}ms")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to list agents: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Agent listing failed: {str(e)}"
        )


@router.get("/{agent_id}", response_model=AgentResponse)
async def get_agent(
    agent_id: str = Path(..., description="Agent ID"),
    orchestrator: SimpleOrchestrator = Depends(get_orchestrator)
):
    """
    Get detailed information about a specific agent.
    
    Retrieves current agent status, performance metrics, and configuration
    from the orchestrator.
    """
    try:
        # Validate UUID format
        try:
            uuid.UUID(agent_id)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="Invalid agent ID format"
            )
        
        # Get agent from orchestrator
        try:
            agent_data = await orchestrator.get_agent(agent_id)
        except Exception as e:
            logger.warning(f"Orchestrator query failed for agent {agent_id}: {e}")
            raise HTTPException(
                status_code=404,
                detail="Agent not found"
            )
        
        if not agent_data:
            raise HTTPException(
                status_code=404,
                detail="Agent not found"
            )
        
        # Convert to response format
        response = AgentResponse(
            id=uuid.UUID(agent_data["id"]),
            name=agent_data["name"],
            type=agent_data.get("type", AgentType.CLAUDE),
            role=agent_data.get("role"),
            capabilities=agent_data.get("capabilities", []),
            status=agent_data.get("status", AgentStatus.UNKNOWN),
            config=agent_data.get("config", {}),
            tmux_session=agent_data.get("tmux_session"),
            total_tasks_completed=agent_data.get("total_tasks_completed", 0),
            total_tasks_failed=agent_data.get("total_tasks_failed", 0),
            average_response_time=agent_data.get("average_response_time", 0.0),
            context_window_usage=agent_data.get("context_window_usage", 0.0),
            created_at=agent_data.get("created_at"),
            updated_at=agent_data.get("updated_at"),
            last_heartbeat=agent_data.get("last_heartbeat"),
            last_active=agent_data.get("last_active")
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get agent {agent_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Agent retrieval failed: {str(e)}"
        )


@router.put("/{agent_id}/status", response_model=AgentOperationResponse)
async def update_agent_status(
    agent_id: str = Path(..., description="Agent ID"),
    status_update: AgentStatusUpdateRequest = ...,
    orchestrator: SimpleOrchestrator = Depends(get_orchestrator)
):
    """
    Update agent status (activate, deactivate, etc.).
    
    Controls agent lifecycle through the orchestrator, enabling proper
    resource management and task assignment control.
    """
    try:
        # Validate UUID format
        try:
            uuid.UUID(agent_id)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="Invalid agent ID format"
            )
        
        # Update status through orchestrator
        try:
            result = await orchestrator.update_agent_status(
                agent_id=agent_id,
                new_status=status_update.status,
                reason=status_update.reason
            )
        except Exception as e:
            logger.warning(f"Orchestrator status update failed for agent {agent_id}: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Status update failed: {str(e)}"
            )
        
        if not result.get("success", False):
            raise HTTPException(
                status_code=400,
                detail=result.get("message", "Status update failed")
            )
        
        response = AgentOperationResponse(
            success=True,
            message=f"Agent status updated to {status_update.status.value}",
            agent_id=agent_id
        )
        
        logger.info(f"✅ Updated agent {agent_id} status to {status_update.status.value}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update agent {agent_id} status: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Status update failed: {str(e)}"
        )


@router.delete("/{agent_id}", response_model=AgentOperationResponse)
async def delete_agent(
    agent_id: str = Path(..., description="Agent ID"),
    force: bool = Query(False, description="Force deletion even if agent is active"),
    orchestrator: SimpleOrchestrator = Depends(get_orchestrator)
):
    """
    Delete an agent from the system.
    
    Removes the agent from the orchestrator and cleans up associated resources.
    Optionally supports force deletion for emergency cleanup.
    """
    try:
        # Validate UUID format
        try:
            uuid.UUID(agent_id)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="Invalid agent ID format"
            )
        
        # Delete agent through orchestrator
        try:
            result = await orchestrator.delete_agent(
                agent_id=agent_id,
                force=force
            )
        except Exception as e:
            logger.warning(f"Orchestrator deletion failed for agent {agent_id}: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Agent deletion failed: {str(e)}"
            )
        
        if not result.get("success", False):
            raise HTTPException(
                status_code=400,
                detail=result.get("message", "Agent deletion failed")
            )
        
        response = AgentOperationResponse(
            success=True,
            message=f"Agent {agent_id} deleted successfully",
            agent_id=agent_id
        )
        
        logger.info(f"✅ Deleted agent {agent_id}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete agent {agent_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Agent deletion failed: {str(e)}"
        )


@router.get("/{agent_id}/stats", response_model=AgentStatsResponse)
async def get_agent_stats(
    agent_id: str = Path(..., description="Agent ID"),
    orchestrator: SimpleOrchestrator = Depends(get_orchestrator)
):
    """
    Get detailed performance statistics for an agent.
    
    Provides comprehensive metrics including task completion rates,
    response times, and resource utilization.
    """
    try:
        # Validate UUID format
        try:
            agent_uuid = uuid.UUID(agent_id)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="Invalid agent ID format"
            )
        
        # Get stats from orchestrator
        try:
            stats = await orchestrator.get_agent_stats(agent_id)
        except Exception as e:
            logger.warning(f"Orchestrator stats query failed for agent {agent_id}: {e}")
            raise HTTPException(
                status_code=404,
                detail="Agent statistics not available"
            )
        
        if not stats:
            raise HTTPException(
                status_code=404,
                detail="Agent not found"
            )
        
        response = AgentStatsResponse(
            agent_id=agent_uuid,
            total_tasks_completed=stats.get("total_tasks_completed", 0),
            total_tasks_failed=stats.get("total_tasks_failed", 0),
            success_rate=stats.get("success_rate", 0.0),
            average_response_time=stats.get("average_response_time", 0.0),
            context_window_usage=stats.get("context_window_usage", 0.0),
            uptime_hours=stats.get("uptime_hours", 0.0),
            last_active=stats.get("last_active"),
            capabilities_count=len(stats.get("capabilities", []))
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get stats for agent {agent_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Statistics retrieval failed: {str(e)}"
        )


# Health check endpoint for agents subsystem
@router.get("/health/status")
async def agents_health_check():
    """Health check endpoint for the agents subsystem."""
    try:
        # Test orchestrator connectivity
        orchestrator_healthy = True
        orchestrator_error = None
        
        try:
            orchestrator = SimpleOrchestrator()
            await orchestrator.initialize()
            # Basic health check operation
            await orchestrator.get_system_health()
        except Exception as e:
            orchestrator_healthy = False
            orchestrator_error = str(e)
        
        health_status = {
            "service": "agents_api",
            "healthy": orchestrator_healthy,
            "timestamp": datetime.utcnow().isoformat(),
            "components": {
                "orchestrator": {
                    "healthy": orchestrator_healthy,
                    "error": orchestrator_error
                }
            }
        }
        
        status_code = 200 if orchestrator_healthy else 503
        return JSONResponse(content=health_status, status_code=status_code)
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            content={
                "service": "agents_api",
                "healthy": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            },
            status_code=503
        )