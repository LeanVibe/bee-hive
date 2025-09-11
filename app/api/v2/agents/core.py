"""
Core Agent Management Endpoints for AgentManagementAPI v2

Consolidates core agent lifecycle management from multiple source modules:
- app/api/endpoints/agents.py -> Core CRUD operations
- app/api/v2/agents.py -> Epic B agent management  
- app/api/agent_activation.py -> Agent activation control
- app/api/v1/agents_simple.py -> Simple agent operations

Provides unified, high-performance agent management with <200ms response times.
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

from fastapi import APIRouter, HTTPException, Depends, Query, Path, BackgroundTasks
from fastapi.responses import JSONResponse
import structlog

from .models import (
    AgentCreateRequest, AgentResponse, AgentListResponse, AgentStatsResponse,
    AgentStatusUpdateRequest, AgentOperationResponse, AgentActivationRequest,
    AgentActivationResponse, SystemStatusResponse, AgentHealthResponse
)
from .middleware import (
    validate_agent_id, validate_pagination, performance_monitor,
    require_agent_permissions, get_authenticated_user
)
try:
    from ....core.database import get_async_session
except ImportError:
    async def get_async_session():
        return None

try:
    from ....core.simple_orchestrator import SimpleOrchestrator, AgentRole, AgentStatus
except ImportError:
    from enum import Enum
    
    class AgentRole(Enum):
        BACKEND_DEVELOPER = "backend_developer"
        FRONTEND_DEVELOPER = "frontend_developer" 
        DEVOPS_ENGINEER = "devops_engineer"
        QA_ENGINEER = "qa_engineer"
    
    class AgentStatus(Enum):
        CREATED = "CREATED"
        ACTIVE = "ACTIVE"
        INACTIVE = "INACTIVE"
        
    class SimpleOrchestrator:
        async def spawn_agent(self, **kwargs):
            return "mock_agent_id"

try:
    from ....core.orchestrator import get_orchestrator as get_production_orchestrator
except ImportError:
    from ....core.simple_orchestrator import get_simple_orchestrator
    def get_production_orchestrator():
        return get_simple_orchestrator()

try:
    from ....models.agent import AgentType
except ImportError:
    from enum import Enum
    
    class AgentType(Enum):
        CLAUDE = "claude"
        OPENAI = "openai"

logger = structlog.get_logger()
router = APIRouter(prefix="/agents", tags=["Agent Management"])

# Global orchestrator instance
_orchestrator: Optional[SimpleOrchestrator] = None

async def get_orchestrator() -> SimpleOrchestrator:
    """Get consolidated orchestrator instance with Epic 1 integration."""
    global _orchestrator
    if _orchestrator is None:
        try:
            # Try to get ConsolidatedProductionOrchestrator first (Epic 1)
            _orchestrator = await get_production_orchestrator()
            logger.info("Using ConsolidatedProductionOrchestrator for agent management")
        except Exception as e:
            logger.warning(f"ConsolidatedProductionOrchestrator unavailable: {e}")
            # Fallback to SimpleOrchestrator
            from app.core.simple_orchestrator import get_simple_orchestrator
            _orchestrator = get_simple_orchestrator()
            await _orchestrator.initialize()
            logger.info("Using SimpleOrchestrator as fallback")
    
    return _orchestrator


# ========================================
# Core Agent Lifecycle Management
# ========================================

@router.post("/", response_model=AgentResponse, status_code=201)
@performance_monitor(target_ms=200)
async def create_agent(
    request: AgentCreateRequest,
    background_tasks: BackgroundTasks,
    user: Dict[str, Any] = Depends(require_agent_permissions),
    orchestrator: SimpleOrchestrator = Depends(get_orchestrator)
) -> AgentResponse:
    """
    Create a new agent with comprehensive lifecycle management.
    
    Consolidates agent creation from multiple sources with:
    - Epic 1 ConsolidatedProductionOrchestrator integration
    - Performance monitoring (<200ms target)
    - Comprehensive error handling and validation
    - Background task initialization
    """
    try:
        start_time = datetime.utcnow()
        logger.info("Creating agent", agent_name=request.name, role=request.role, user_id=user.get('user_id'))
        
        # Validate and map role
        try:
            agent_role = AgentRole(request.role)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid role. Valid roles: {[r.value for r in AgentRole]}"
            )
        
        # Generate agent configuration
        agent_config = {
            "id": str(uuid.uuid4()),
            "name": request.name,
            "role": agent_role,
            "type": AgentType.CLAUDE if request.type == "claude_code" else AgentType.OPENAI,
            "capabilities": request.capabilities,
            "system_prompt": request.system_prompt,
            "config": request.config or {},
            "status": AgentStatus.CREATED,
            "created_by": user.get('user_id'),
            "workspace_name": request.workspace_name,
            "git_branch": request.git_branch
        }
        
        # Create agent using orchestrator
        try:
            agent_id = await orchestrator.spawn_agent(
                role=agent_role,
                task_id=request.task_id,
                workspace_name=request.workspace_name,
                git_branch=request.git_branch,
                agent_config=agent_config
            )
        except Exception as e:
            logger.error("Orchestrator agent creation failed", error=str(e))
            raise HTTPException(
                status_code=500,
                detail=f"Agent creation failed: {str(e)}"
            )
        
        # Get created agent details
        agent_data = await orchestrator.get_agent_details(agent_id)
        if not agent_data:
            raise HTTPException(
                status_code=500,
                detail="Agent created but details not retrievable"
            )
        
        # Prepare response
        response = AgentResponse(
            id=agent_id,
            name=request.name,
            role=request.role,
            type=request.type,
            status=AgentStatus.CREATED.value,
            capabilities=request.capabilities,
            config=request.config or {},
            created_at=start_time,
            updated_at=start_time,
            current_task_id=request.task_id
        )
        
        # Add background initialization task
        background_tasks.add_task(
            _initialize_agent_background,
            agent_id,
            orchestrator,
            user.get('user_id')
        )
        
        # Performance monitoring
        elapsed_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        logger.info(
            "Agent created successfully",
            agent_id=agent_id,
            elapsed_ms=round(elapsed_ms, 2),
            user_id=user.get('user_id')
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Agent creation failed", error=str(e), user_id=user.get('user_id'))
        raise HTTPException(
            status_code=500,
            detail=f"Internal error during agent creation: {str(e)}"
        )


@router.get("/", response_model=AgentListResponse)
@performance_monitor(target_ms=150)
async def list_agents(
    status: Optional[str] = Query(None, description="Filter by agent status"),
    role: Optional[str] = Query(None, description="Filter by agent role"),
    limit: int = Query(50, ge=1, le=100, description="Maximum agents to return"),
    offset: int = Query(0, ge=0, description="Number of agents to skip"),
    user: Dict[str, Any] = Depends(require_agent_permissions),
    orchestrator: SimpleOrchestrator = Depends(get_orchestrator)
) -> AgentListResponse:
    """
    List agents with advanced filtering and pagination.
    
    Consolidates agent listing from multiple sources with:
    - High-performance querying (<150ms target)
    - Advanced filtering by status and role
    - Pagination with performance optimization
    - Real-time status updates
    """
    try:
        start_time = datetime.utcnow()
        limit, offset = validate_pagination(limit, offset)
        
        # Get agents from orchestrator
        agents_data = await orchestrator.list_agents(
            status_filter=status,
            role_filter=role,
            limit=limit,
            offset=offset,
            user_context=user
        )
        
        # Convert to response format
        agents = []
        for agent_data in agents_data.get('agents', []):
            try:
                agent_response = AgentResponse(
                    id=agent_data['id'],
                    name=agent_data.get('name', f"Agent-{agent_data['id'][:8]}"),
                    role=agent_data.get('role', 'unknown'),
                    type=agent_data.get('type', 'claude_code'),
                    status=agent_data.get('status', 'unknown'),
                    capabilities=agent_data.get('capabilities', []),
                    config=agent_data.get('config', {}),
                    created_at=agent_data.get('created_at', datetime.utcnow()),
                    updated_at=agent_data.get('updated_at'),
                    last_active=agent_data.get('last_active'),
                    current_task_id=agent_data.get('current_task_id'),
                    total_tasks_completed=agent_data.get('total_tasks_completed', 0),
                    total_tasks_failed=agent_data.get('total_tasks_failed', 0),
                    average_response_time=agent_data.get('average_response_time', 0.0)
                )
                agents.append(agent_response)
            except Exception as e:
                logger.warning("Skipping invalid agent data", agent_id=agent_data.get('id'), error=str(e))
                continue
        
        # Calculate statistics
        stats = agents_data.get('stats', {})
        response = AgentListResponse(
            agents=agents,
            total=stats.get('total', len(agents)),
            active=stats.get('active', 0),
            inactive=stats.get('inactive', 0),
            offset=offset,
            limit=limit
        )
        
        # Performance monitoring
        elapsed_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        logger.info(
            "Agent list retrieved",
            count=len(agents),
            elapsed_ms=round(elapsed_ms, 2),
            user_id=user.get('user_id')
        )
        
        return response
        
    except Exception as e:
        logger.error("Agent listing failed", error=str(e), user_id=user.get('user_id'))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve agent list: {str(e)}"
        )


@router.get("/{agent_id}", response_model=AgentResponse)
@performance_monitor(target_ms=100)
async def get_agent(
    agent_id: str = Path(..., description="Agent ID"),
    user: Dict[str, Any] = Depends(require_agent_permissions),
    orchestrator: SimpleOrchestrator = Depends(get_orchestrator)
) -> AgentResponse:
    """
    Get detailed agent information with real-time status.
    
    Provides comprehensive agent details with:
    - Real-time status and performance metrics
    - Task assignment information
    - Resource usage statistics
    - Health monitoring data
    """
    try:
        agent_id = validate_agent_id(agent_id)
        
        # Get agent details from orchestrator
        agent_data = await orchestrator.get_agent_details(agent_id)
        if not agent_data:
            raise HTTPException(
                status_code=404,
                detail="Agent not found"
            )
        
        # Convert to response format
        response = AgentResponse(
            id=agent_data['id'],
            name=agent_data.get('name', f"Agent-{agent_data['id'][:8]}"),
            role=agent_data.get('role', 'unknown'),
            type=agent_data.get('type', 'claude_code'),
            status=agent_data.get('status', 'unknown'),
            capabilities=agent_data.get('capabilities', []),
            config=agent_data.get('config', {}),
            created_at=agent_data.get('created_at', datetime.utcnow()),
            updated_at=agent_data.get('updated_at'),
            last_active=agent_data.get('last_active'),
            last_heartbeat=agent_data.get('last_heartbeat'),
            current_task_id=agent_data.get('current_task_id'),
            total_tasks_completed=agent_data.get('total_tasks_completed', 0),
            total_tasks_failed=agent_data.get('total_tasks_failed', 0),
            average_response_time=agent_data.get('average_response_time', 0.0),
            context_window_usage=agent_data.get('context_window_usage', 0.0),
            tmux_session=agent_data.get('tmux_session')
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Agent retrieval failed", agent_id=agent_id, error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve agent details: {str(e)}"
        )


@router.put("/{agent_id}/status", response_model=AgentOperationResponse)
@performance_monitor(target_ms=200)
async def update_agent_status(
    agent_id: str = Path(..., description="Agent ID"),
    request: AgentStatusUpdateRequest = ...,
    user: Dict[str, Any] = Depends(require_agent_permissions),
    orchestrator: SimpleOrchestrator = Depends(get_orchestrator)
) -> AgentOperationResponse:
    """
    Update agent status with comprehensive lifecycle management.
    
    Supports all agent status transitions with:
    - Validation of status transitions
    - Proper resource cleanup
    - Event notification
    - Audit logging
    """
    try:
        agent_id = validate_agent_id(agent_id)
        
        # Validate status transition
        try:
            new_status = AgentStatus(request.status)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid status. Valid statuses: {[s.value for s in AgentStatus]}"
            )
        
        # Update status through orchestrator
        result = await orchestrator.update_agent_status(
            agent_id=agent_id,
            new_status=new_status,
            reason=request.reason,
            user_context=user
        )
        
        if not result.get('success', False):
            raise HTTPException(
                status_code=400,
                detail=result.get('message', 'Status update failed')
            )
        
        response = AgentOperationResponse(
            success=True,
            message=f"Agent status updated to {new_status.value}",
            agent_id=agent_id,
            operation_details={
                'old_status': result.get('old_status'),
                'new_status': new_status.value,
                'reason': request.reason,
                'updated_by': user.get('user_id')
            }
        )
        
        logger.info(
            "Agent status updated",
            agent_id=agent_id,
            new_status=new_status.value,
            user_id=user.get('user_id')
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Agent status update failed", agent_id=agent_id, error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Status update failed: {str(e)}"
        )


@router.delete("/{agent_id}", response_model=AgentOperationResponse)
@performance_monitor(target_ms=300)
async def delete_agent(
    agent_id: str = Path(..., description="Agent ID"),
    force: bool = Query(False, description="Force deletion even if agent is active"),
    user: Dict[str, Any] = Depends(require_agent_permissions),
    orchestrator: SimpleOrchestrator = Depends(get_orchestrator)
) -> AgentOperationResponse:
    """
    Delete agent with comprehensive cleanup.
    
    Performs complete agent deletion with:
    - Graceful shutdown procedures
    - Resource cleanup
    - Task reassignment if needed
    - Comprehensive audit logging
    """
    try:
        agent_id = validate_agent_id(agent_id)
        
        # Delete agent through orchestrator
        result = await orchestrator.delete_agent(
            agent_id=agent_id,
            force=force,
            user_context=user
        )
        
        if not result.get('success', False):
            raise HTTPException(
                status_code=400,
                detail=result.get('message', 'Agent deletion failed')
            )
        
        response = AgentOperationResponse(
            success=True,
            message=f"Agent {agent_id} deleted successfully",
            agent_id=agent_id,
            operation_details={
                'forced': force,
                'cleanup_performed': result.get('cleanup_performed', []),
                'deleted_by': user.get('user_id')
            }
        )
        
        logger.info(
            "Agent deleted",
            agent_id=agent_id,
            force=force,
            user_id=user.get('user_id')
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Agent deletion failed", agent_id=agent_id, error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Agent deletion failed: {str(e)}"
        )


# ========================================
# Agent System Control
# ========================================

@router.post("/activate", response_model=AgentActivationResponse)
@performance_monitor(target_ms=5000)  # System activation can take up to 5s
async def activate_agent_system(
    request: AgentActivationRequest,
    background_tasks: BackgroundTasks,
    user: Dict[str, Any] = Depends(require_agent_permissions),
    orchestrator: SimpleOrchestrator = Depends(get_orchestrator)
) -> AgentActivationResponse:
    """
    Activate the multi-agent system with development team.
    
    Transforms the system from infrastructure-only to operational with:
    - Core development team spawning
    - System readiness validation
    - Performance benchmarking
    - Automatic task assignment if requested
    """
    try:
        start_time = datetime.utcnow()
        logger.info(
            "Activating agent system",
            team_size=request.team_size,
            auto_start=request.auto_start_tasks,
            user_id=user.get('user_id')
        )
        
        # Define core roles to activate
        roles_to_activate = request.roles or [
            'backend_developer',
            'frontend_developer', 
            'devops_engineer',
            'qa_engineer'
        ]
        
        # Spawn development team
        team_composition = {}
        active_agents = {}
        
        for role in roles_to_activate[:request.team_size]:
            try:
                agent_role = AgentRole(role)
                agent_id = await orchestrator.spawn_agent(
                    role=agent_role,
                    workspace_config=request.workspace_config,
                    user_context=user
                )
                
                team_composition[role] = agent_id
                agent_details = await orchestrator.get_agent_details(agent_id)
                active_agents[agent_id] = agent_details
                
                logger.info(f"Activated {role} agent", agent_id=agent_id)
                
            except Exception as e:
                logger.warning(f"Failed to activate {role} agent", error=str(e))
                team_composition[role] = f"failed: {str(e)}"
        
        # Start background tasks if requested
        if request.auto_start_tasks:
            background_tasks.add_task(
                _start_demonstration_tasks,
                list(active_agents.keys()),
                orchestrator,
                user.get('user_id')
            )
        
        # Calculate activation time
        activation_time = (datetime.utcnow() - start_time).total_seconds()
        
        response = AgentActivationResponse(
            success=len(active_agents) > 0,
            message=f"Successfully activated {len(active_agents)} agents",
            active_agents=active_agents,
            team_composition=team_composition,
            activation_time=activation_time,
            system_status="ready" if len(active_agents) >= 2 else "partial"
        )
        
        logger.info(
            "Agent system activation completed",
            active_count=len(active_agents),
            activation_time=activation_time,
            user_id=user.get('user_id')
        )
        
        return response
        
    except Exception as e:
        logger.error("Agent system activation failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"System activation failed: {str(e)}"
        )


@router.get("/status", response_model=SystemStatusResponse)
@performance_monitor(target_ms=50)
async def get_agent_system_status(
    orchestrator: SimpleOrchestrator = Depends(get_orchestrator)
) -> SystemStatusResponse:
    """
    Get comprehensive agent system status.
    
    Provides real-time system status including:
    - Agent counts and status distribution
    - System readiness indicators
    - Performance metrics
    - Capability overview
    """
    try:
        # Get system status from orchestrator
        status_data = await orchestrator.get_system_status()
        
        agent_details = status_data.get('agents', {}).get('details', {})
        total_agents = len(agent_details)
        
        # Calculate system capabilities
        capabilities_summary = set()
        for agent_data in agent_details.values():
            capabilities_summary.update(agent_data.get('capabilities', []))
        
        response = SystemStatusResponse(
            active=total_agents > 0,
            agent_count=total_agents,
            system_ready=total_agents >= 2,
            orchestrator_type=status_data.get('orchestrator_type', 'ConsolidatedProductionOrchestrator'),
            orchestrator_health=status_data.get('health', 'healthy'),
            performance=status_data.get('performance', {}),
            agents=agent_details,
            capabilities_summary=list(capabilities_summary)
        )
        
        return response
        
    except Exception as e:
        logger.error("System status retrieval failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get system status: {str(e)}"
        )


@router.get("/{agent_id}/health", response_model=AgentHealthResponse)
@performance_monitor(target_ms=100)
async def get_agent_health(
    agent_id: str = Path(..., description="Agent ID"),
    user: Dict[str, Any] = Depends(require_agent_permissions),
    orchestrator: SimpleOrchestrator = Depends(get_orchestrator)
) -> AgentHealthResponse:
    """
    Get comprehensive agent health status.
    
    Provides detailed health metrics including:
    - Operational status and uptime
    - Resource usage and performance
    - Task execution health
    - System integration status
    """
    try:
        agent_id = validate_agent_id(agent_id)
        
        # Get health data from orchestrator
        health_data = await orchestrator.get_agent_health(agent_id)
        if not health_data:
            raise HTTPException(
                status_code=404,
                detail="Agent not found or health data unavailable"
            )
        
        response = AgentHealthResponse(
            agent_id=agent_id,
            status=health_data.get('status', 'unknown'),
            last_activity=health_data.get('last_activity', datetime.utcnow().isoformat()),
            uptime_seconds=health_data.get('uptime_seconds', 0.0),
            current_task=health_data.get('current_task'),
            healthy=health_data.get('healthy', False),
            session=health_data.get('session'),
            performance_metrics=health_data.get('performance_metrics', {}),
            resource_usage=health_data.get('resource_usage', {})
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Agent health check failed", agent_id=agent_id, error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Health check failed: {str(e)}"
        )


@router.get("/{agent_id}/stats", response_model=AgentStatsResponse)
@performance_monitor(target_ms=150)
async def get_agent_stats(
    agent_id: str = Path(..., description="Agent ID"),
    user: Dict[str, Any] = Depends(require_agent_permissions),
    orchestrator: SimpleOrchestrator = Depends(get_orchestrator)
) -> AgentStatsResponse:
    """
    Get detailed agent performance statistics.
    
    Provides comprehensive analytics including:
    - Task completion rates and performance trends
    - Response time analysis and optimization insights
    - Resource utilization patterns
    - Capability effectiveness metrics
    """
    try:
        agent_id = validate_agent_id(agent_id)
        
        # Get statistics from orchestrator
        stats_data = await orchestrator.get_agent_stats(agent_id)
        if not stats_data:
            raise HTTPException(
                status_code=404,
                detail="Agent not found or statistics unavailable"
            )
        
        # Calculate success rate
        total_completed = stats_data.get('total_tasks_completed', 0)
        total_failed = stats_data.get('total_tasks_failed', 0)
        success_rate = (
            total_completed / (total_completed + total_failed) * 100
            if (total_completed + total_failed) > 0 else 0.0
        )
        
        response = AgentStatsResponse(
            agent_id=agent_id,
            total_tasks_completed=total_completed,
            total_tasks_failed=total_failed,
            success_rate=success_rate,
            average_response_time=stats_data.get('average_response_time', 0.0),
            context_window_usage=stats_data.get('context_window_usage', 0.0),
            uptime_hours=stats_data.get('uptime_hours', 0.0),
            last_active=stats_data.get('last_active'),
            capabilities_count=len(stats_data.get('capabilities', [])),
            performance_trend=stats_data.get('performance_trend', {})
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Agent statistics retrieval failed", agent_id=agent_id, error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Statistics retrieval failed: {str(e)}"
        )


# ========================================
# Background Task Functions
# ========================================

async def _initialize_agent_background(
    agent_id: str,
    orchestrator: SimpleOrchestrator,
    user_id: str
) -> None:
    """Background task to initialize agent after creation."""
    try:
        await asyncio.sleep(2)  # Allow creation to complete
        
        # Perform agent initialization
        await orchestrator.initialize_agent(agent_id)
        
        # Update agent status to active
        await orchestrator.update_agent_status(
            agent_id=agent_id,
            new_status=AgentStatus.ACTIVE,
            reason="Background initialization completed"
        )
        
        logger.info("Agent initialization completed", agent_id=agent_id, user_id=user_id)
        
    except Exception as e:
        logger.error("Agent background initialization failed", agent_id=agent_id, error=str(e))


async def _start_demonstration_tasks(
    agent_ids: List[str],
    orchestrator: SimpleOrchestrator,
    user_id: str
) -> None:
    """Background task to start demonstration tasks for activated agents."""
    try:
        await asyncio.sleep(5)  # Wait for agents to fully initialize
        
        demo_tasks = [
            ("System architecture review", "backend_developer"),
            ("UI/UX optimization analysis", "frontend_developer"),
            ("Infrastructure health check", "devops_engineer"),
            ("Quality assurance validation", "qa_engineer")
        ]
        
        for agent_id in agent_ids:
            agent_data = await orchestrator.get_agent_details(agent_id)
            if not agent_data:
                continue
                
            agent_role = agent_data.get('role', '')
            
            # Find appropriate demo task
            demo_task = None
            for task_desc, role in demo_tasks:
                if role in agent_role.lower():
                    demo_task = task_desc
                    break
            
            if demo_task:
                task_id = f"demo_{agent_id[:8]}_{int(datetime.utcnow().timestamp())}"
                await orchestrator.assign_task(agent_id, task_id, {
                    'description': demo_task,
                    'type': 'demonstration',
                    'priority': 'low',
                    'auto_complete': True
                })
                
                logger.info("Demo task assigned", agent_id=agent_id, task=demo_task)
        
    except Exception as e:
        logger.error("Demonstration tasks initialization failed", error=str(e), user_id=user_id)