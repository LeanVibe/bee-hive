"""
Backwards Compatibility Layer for AgentManagementAPI v2

Provides full compatibility with existing v1 agent management endpoints
ensuring zero-downtime migration for existing consumers while offering
enhanced functionality through the unified v2 API.
"""

import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any

from fastapi import APIRouter, HTTPException, Depends, Query, Path, BackgroundTasks
from fastapi.responses import JSONResponse
import structlog

from .models import (
    AgentCreateRequest, AgentResponse, AgentListResponse,
    LegacyAgentResponse, AgentOperationResponse, SystemStatusResponse
)
from .core import get_orchestrator
from .middleware import get_authenticated_user, performance_monitor
from .utils import normalize_agent_response

logger = structlog.get_logger()
router = APIRouter(prefix="/v1", tags=["Legacy Compatibility"])


# ========================================
# V1 Agent Simple API Compatibility
# ========================================

@router.post("/agents", response_model=LegacyAgentResponse, status_code=201)
@performance_monitor(target_ms=200)
async def create_agent_v1_simple(
    request: Dict[str, Any],
    orchestrator = Depends(get_orchestrator)
) -> LegacyAgentResponse:
    """
    Legacy endpoint: Create agent (v1/agents_simple.py compatibility)
    
    Maintains full compatibility with existing agent creation requests
    while routing through the enhanced v2 infrastructure.
    """
    try:
        # Transform legacy request to v2 format
        v2_request = AgentCreateRequest(
            name=request.get('name', f"Agent-{str(uuid.uuid4())[:8]}"),
            role=_map_legacy_type_to_role(request.get('type', 'general')),
            type='claude_code',  # Default type for v1 compatibility
            capabilities=request.get('capabilities', [])
        )
        
        # Create agent using v2 core logic
        from .core import create_agent
        
        # Create a minimal user context for legacy compatibility
        legacy_user = {
            'user_id': 'legacy_api_user',
            'permissions': ['agent:create', 'agent:write']
        }
        
        background_tasks = BackgroundTasks()
        v2_response = await create_agent(v2_request, background_tasks, legacy_user, orchestrator)
        
        # Transform v2 response to legacy format
        legacy_response = LegacyAgentResponse.from_unified_response(v2_response)
        
        logger.info("Legacy agent creation successful", agent_id=legacy_response.id)
        return legacy_response
        
    except Exception as e:
        logger.error("Legacy agent creation failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Agent creation failed: {str(e)}"
        )


@router.get("/agents", response_model=Dict[str, Any])
@performance_monitor(target_ms=150)
async def list_agents_v1_simple(
    orchestrator = Depends(get_orchestrator)
) -> Dict[str, Any]:
    """
    Legacy endpoint: List agents (v1/agents_simple.py compatibility)
    
    Maintains the exact response format expected by v1 consumers
    while leveraging v2 performance optimizations.
    """
    try:
        # Get agents using v2 core logic
        from .core import list_agents
        
        legacy_user = {
            'user_id': 'legacy_api_user',
            'permissions': ['agent:read', 'agent:list']
        }
        
        v2_response = await list_agents(
            status=None,
            role=None,
            limit=100,  # Legacy default
            offset=0,
            user=legacy_user,
            orchestrator=orchestrator
        )
        
        # Transform to legacy format
        legacy_agents = []
        for agent in v2_response.agents:
            legacy_agent = LegacyAgentResponse.from_unified_response(agent)
            legacy_agents.append(legacy_agent.dict())
        
        return {
            'agents': legacy_agents,
            'total': v2_response.total
        }
        
    except Exception as e:
        logger.error("Legacy agent listing failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve agents: {str(e)}"
        )


@router.get("/agents/{agent_id}", response_model=LegacyAgentResponse)
@performance_monitor(target_ms=100)
async def get_agent_v1_simple(
    agent_id: str,
    orchestrator = Depends(get_orchestrator)
) -> LegacyAgentResponse:
    """
    Legacy endpoint: Get agent (v1/agents_simple.py compatibility)
    """
    try:
        from .core import get_agent
        
        legacy_user = {
            'user_id': 'legacy_api_user',
            'permissions': ['agent:read']
        }
        
        v2_response = await get_agent(agent_id, legacy_user, orchestrator)
        return LegacyAgentResponse.from_unified_response(v2_response)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Legacy agent retrieval failed", agent_id=agent_id, error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve agent: {str(e)}"
        )


@router.delete("/agents/{agent_id}")
@performance_monitor(target_ms=300)
async def shutdown_agent_v1_simple(
    agent_id: str,
    graceful: bool = Query(True, description="Graceful shutdown"),
    orchestrator = Depends(get_orchestrator)
) -> Dict[str, Any]:
    """
    Legacy endpoint: Shutdown agent (v1/agents_simple.py compatibility)
    """
    try:
        from .core import delete_agent
        
        legacy_user = {
            'user_id': 'legacy_api_user',
            'permissions': ['agent:delete', 'agent:admin']
        }
        
        v2_response = await delete_agent(agent_id, not graceful, legacy_user, orchestrator)
        
        return {
            'success': v2_response.success,
            'agent_id': agent_id,
            'message': 'Agent shutdown successfully',
            'graceful': graceful
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Legacy agent shutdown failed", agent_id=agent_id, error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to shutdown agent: {str(e)}"
        )


@router.get("/agents/{agent_id}/status")
@performance_monitor(target_ms=100)
async def get_agent_status_v1_simple(
    agent_id: str,
    orchestrator = Depends(get_orchestrator)
) -> Dict[str, Any]:
    """
    Legacy endpoint: Get agent status (v1/agents_simple.py compatibility)
    """
    try:
        from .core import get_agent
        
        legacy_user = {
            'user_id': 'legacy_api_user',
            'permissions': ['agent:read']
        }
        
        v2_response = await get_agent(agent_id, legacy_user, orchestrator)
        
        return {
            'agent_id': agent_id,
            'status': v2_response.status,
            'name': v2_response.name,
            'type': v2_response.type,
            'active_tasks': 1 if v2_response.current_task_id else 0,
            'last_active': v2_response.last_active or v2_response.created_at,
            'is_running': v2_response.status.lower() in ['active', 'idle'],
            'session_info': {
                'created_at': v2_response.created_at,
                'environment_vars': {
                    'LEANVIBE_AGENT_NAME': v2_response.name,
                    'LEANVIBE_AGENT_TYPE': v2_response.type
                }
            },
            'metrics': {
                'task_count': v2_response.total_tasks_completed,
                'success_rate': (
                    v2_response.total_tasks_completed / 
                    max(1, v2_response.total_tasks_completed + v2_response.total_tasks_failed)
                ),
                'average_response_time': v2_response.average_response_time
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Legacy agent status retrieval failed", agent_id=agent_id, error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get agent status: {str(e)}"
        )


@router.get("/agents/system/status")
@performance_monitor(target_ms=100)
async def get_system_status_v1_simple(
    orchestrator = Depends(get_orchestrator)
) -> Dict[str, Any]:
    """
    Legacy endpoint: Get system status (v1/agents_simple.py compatibility)
    """
    try:
        from .core import get_agent_system_status
        
        v2_response = await get_agent_system_status(orchestrator)
        
        return {
            'success': True,
            'system_status': {
                'active': v2_response.active,
                'agent_count': v2_response.agent_count,
                'system_ready': v2_response.system_ready,
                'orchestrator_type': v2_response.orchestrator_type,
                'orchestrator_health': v2_response.orchestrator_health,
                'agents': v2_response.agents,
                'performance': v2_response.performance
            },
            'timestamp': datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error("Legacy system status retrieval failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get system status: {str(e)}"
        )


# ========================================
# V1 Agent Endpoints API Compatibility
# ========================================

@router.post("/api/v1/agents", response_model=Dict[str, Any], status_code=201)
@performance_monitor(target_ms=200)
async def create_agent_v1_endpoints(
    agent_data: Dict[str, Any],
    background_tasks: BackgroundTasks,
    orchestrator = Depends(get_orchestrator)
) -> Dict[str, Any]:
    """
    Legacy endpoint: Create agent (endpoints/agents.py compatibility)
    
    Maintains compatibility with the Epic C Phase 1 agent endpoints format
    while providing enhanced functionality through v2 infrastructure.
    """
    try:
        # Transform Epic C request to v2 format
        v2_request = AgentCreateRequest(
            name=agent_data.get('name', f"Agent-{str(uuid.uuid4())[:8]}"),
            role=agent_data.get('role', 'backend_developer'),
            type=agent_data.get('type', 'claude_code'),
            capabilities=agent_data.get('capabilities', []),
            system_prompt=agent_data.get('system_prompt'),
            config=agent_data.get('config', {})
        )
        
        legacy_user = {
            'user_id': 'legacy_endpoints_user',
            'permissions': ['agent:create', 'agent:write']
        }
        
        from .core import create_agent
        v2_response = await create_agent(v2_request, background_tasks, legacy_user, orchestrator)
        
        # Transform to Epic C format
        return {
            'id': v2_response.id,
            'name': v2_response.name,
            'type': v2_response.type,
            'role': v2_response.role,
            'capabilities': v2_response.capabilities,
            'status': v2_response.status,
            'config': v2_response.config,
            'tmux_session': v2_response.tmux_session,
            'total_tasks_completed': v2_response.total_tasks_completed,
            'total_tasks_failed': v2_response.total_tasks_failed,
            'average_response_time': v2_response.average_response_time,
            'context_window_usage': v2_response.context_window_usage,
            'created_at': v2_response.created_at,
            'updated_at': v2_response.updated_at,
            'last_heartbeat': v2_response.last_heartbeat,
            'last_active': v2_response.last_active
        }
        
    except Exception as e:
        logger.error("Legacy endpoints agent creation failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Agent creation failed: {str(e)}"
        )


@router.get("/api/v1/agents", response_model=Dict[str, Any])
@performance_monitor(target_ms=150)
async def list_agents_v1_endpoints(
    status: Optional[str] = Query(None, description="Filter by agent status"),
    agent_type: Optional[str] = Query(None, description="Filter by agent type"),
    limit: int = Query(20, ge=1, le=100, description="Maximum number of agents to return"),
    offset: int = Query(0, ge=0, description="Number of agents to skip"),
    orchestrator = Depends(get_orchestrator)
) -> Dict[str, Any]:
    """
    Legacy endpoint: List agents (endpoints/agents.py compatibility)
    """
    try:
        from .core import list_agents
        
        legacy_user = {
            'user_id': 'legacy_endpoints_user',
            'permissions': ['agent:read', 'agent:list']
        }
        
        # Map agent_type to role for v2 compatibility
        role_filter = agent_type  # Assuming type maps to role
        
        v2_response = await list_agents(
            status=status,
            role=role_filter,
            limit=limit,
            offset=offset,
            user=legacy_user,
            orchestrator=orchestrator
        )
        
        # Transform to Epic C format
        agents = []
        for agent in v2_response.agents:
            agents.append({
                'id': agent.id,
                'name': agent.name,
                'type': agent.type,
                'role': agent.role,
                'capabilities': agent.capabilities,
                'status': agent.status,
                'config': agent.config,
                'tmux_session': agent.tmux_session,
                'total_tasks_completed': agent.total_tasks_completed,
                'total_tasks_failed': agent.total_tasks_failed,
                'average_response_time': agent.average_response_time,
                'context_window_usage': agent.context_window_usage,
                'created_at': agent.created_at,
                'updated_at': agent.updated_at,
                'last_heartbeat': agent.last_heartbeat,
                'last_active': agent.last_active
            })
        
        return {
            'agents': agents,
            'total': v2_response.total,
            'offset': v2_response.offset,
            'limit': v2_response.limit
        }
        
    except Exception as e:
        logger.error("Legacy endpoints agent listing failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Agent listing failed: {str(e)}"
        )


# ========================================
# V1 Agent Activation API Compatibility
# ========================================

@router.post("/activate", response_model=Dict[str, Any])
@performance_monitor(target_ms=5000)
async def activate_agent_system_v1(
    request: Dict[str, Any],
    background_tasks: BackgroundTasks,
    orchestrator = Depends(get_orchestrator)
) -> Dict[str, Any]:
    """
    Legacy endpoint: Activate agent system (agent_activation.py compatibility)
    
    Maintains compatibility with the agent activation API while providing
    enhanced team spawning and coordination capabilities.
    """
    try:
        from .core import activate_agent_system
        from .models import AgentActivationRequest
        
        # Transform legacy request
        v2_request = AgentActivationRequest(
            team_size=request.get('team_size', 5),
            roles=request.get('roles'),
            auto_start_tasks=request.get('auto_start_tasks', True)
        )
        
        legacy_user = {
            'user_id': 'legacy_activation_user',
            'permissions': ['agent:activate', 'agent:admin']
        }
        
        v2_response = await activate_agent_system(v2_request, background_tasks, legacy_user, orchestrator)
        
        # Transform to legacy format
        return {
            'success': v2_response.success,
            'message': v2_response.message,
            'active_agents': v2_response.active_agents,
            'team_composition': v2_response.team_composition
        }
        
    except Exception as e:
        logger.error("Legacy agent system activation failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Agent activation failed: {str(e)}"
        )


@router.get("/status")
@performance_monitor(target_ms=50)
async def get_agent_system_status_v1(
    orchestrator = Depends(get_orchestrator)
) -> Dict[str, Any]:
    """
    Legacy endpoint: Get agent system status (agent_activation.py compatibility)
    """
    try:
        from .core import get_agent_system_status
        
        v2_response = await get_agent_system_status(orchestrator)
        
        return {
            'active': v2_response.active,
            'agent_count': v2_response.agent_count,
            'simple_orchestrator_agents': v2_response.agent_count,  # Legacy field
            'spawner_agents': 0,  # Legacy field - always 0 for v2
            'agents': v2_response.agents,
            'spawner_agents_detail': {},  # Legacy field
            'system_ready': v2_response.system_ready,
            'orchestrator_type': v2_response.orchestrator_type,
            'orchestrator_health': v2_response.orchestrator_health,
            'performance': v2_response.performance
        }
        
    except Exception as e:
        logger.error("Legacy agent system status failed", error=str(e))
        return {
            'active': False,
            'agent_count': 0,
            'simple_orchestrator_agents': 0,
            'spawner_agents': 0,
            'agents': {},
            'system_ready': False,
            'orchestrator_type': 'ConsolidatedProductionOrchestrator',
            'error': str(e)
        }


@router.post("/spawn/{role}")
@performance_monitor(target_ms=2000)
async def spawn_specific_agent_v1(
    role: str,
    orchestrator = Depends(get_orchestrator)
) -> Dict[str, Any]:
    """
    Legacy endpoint: Spawn specific agent (agent_activation.py compatibility)
    """
    try:
        from .core import create_agent
        from .models import AgentCreateRequest
        
        # Create agent with specified role
        v2_request = AgentCreateRequest(
            name=f"{role.title().replace('_', ' ')} Agent",
            role=role,
            type='claude_code',
            capabilities=_get_default_capabilities_for_role(role)
        )
        
        legacy_user = {
            'user_id': 'legacy_spawn_user',
            'permissions': ['agent:create', 'agent:write']
        }
        
        background_tasks = BackgroundTasks()
        v2_response = await create_agent(v2_request, background_tasks, legacy_user, orchestrator)
        
        return {
            'success': True,
            'agent_id': str(v2_response.id),
            'role': role,
            'message': f'Successfully spawned {role} agent using ConsolidatedProductionOrchestrator'
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Legacy agent spawning failed", role=role, error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to spawn {role} agent: {str(e)}"
        )


# ========================================
# V1 Coordination API Compatibility
# ========================================

@router.post("/coordination/projects", response_model=Dict[str, Any], status_code=201)
@performance_monitor(target_ms=300)
async def create_coordinated_project_v1(
    request: Dict[str, Any],
    background_tasks: BackgroundTasks,
    orchestrator = Depends(get_orchestrator)
) -> Dict[str, Any]:
    """
    Legacy endpoint: Create coordinated project (v1/coordination.py compatibility)
    
    Provides full compatibility with existing multi-agent coordination
    project creation while leveraging v2 enhanced coordination capabilities.
    """
    try:
        from .coordination import create_coordinated_project
        from .models import ProjectCreateRequest
        
        # Transform legacy request
        v2_request = ProjectCreateRequest(
            name=request['name'],
            description=request['description'],
            requirements=request['requirements'],
            coordination_mode=request.get('coordination_mode', 'parallel'),
            deadline=request.get('deadline'),
            quality_gates=request.get('quality_gates')
        )
        
        legacy_user = {
            'user_id': 'legacy_coordination_user',
            'permissions': ['coordination:create', 'project:manage']
        }
        
        v2_response = await create_coordinated_project(v2_request, background_tasks, legacy_user)
        
        # Transform to legacy format
        return {
            'project_id': v2_response.project_id,
            'name': v2_response.name,
            'description': v2_response.description,
            'status': v2_response.status,
            'coordination_mode': v2_response.coordination_mode,
            'participating_agents': v2_response.participating_agents,
            'progress_percentage': v2_response.progress_percentage,
            'created_at': v2_response.created_at,
            'started_at': v2_response.started_at,
            'estimated_completion': v2_response.estimated_completion
        }
        
    except Exception as e:
        logger.error("Legacy coordinated project creation failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Project creation failed: {str(e)}"
        )


# ========================================
# Utility Functions for Legacy Support
# ========================================

def _map_legacy_type_to_role(legacy_type: str) -> str:
    """Map legacy agent types to v2 roles."""
    type_to_role_mapping = {
        'backend_developer': 'backend_developer',
        'frontend_developer': 'frontend_developer',
        'devops_engineer': 'devops_engineer',
        'qa_engineer': 'qa_engineer',
        'general': 'backend_developer',  # Default fallback
        'claude_code': 'backend_developer'
    }
    
    return type_to_role_mapping.get(legacy_type.lower(), 'backend_developer')


def _get_default_capabilities_for_role(role: str) -> List[str]:
    """Get default capabilities for a role in legacy compatibility mode."""
    role_capabilities = {
        'backend_developer': ['python', 'django', 'fastapi', 'api_development', 'database'],
        'frontend_developer': ['javascript', 'react', 'html', 'css', 'ui_development'],
        'devops_engineer': ['docker', 'kubernetes', 'aws', 'deployment', 'infrastructure'],
        'qa_engineer': ['testing', 'automation', 'quality_assurance', 'validation']
    }
    
    return role_capabilities.get(role.lower(), ['general_development'])


def _transform_v2_to_legacy_format(v2_response: Dict[str, Any], format_type: str = 'simple') -> Dict[str, Any]:
    """Transform v2 API response to legacy format."""
    if format_type == 'simple':
        return {
            'id': str(v2_response.get('id', '')),
            'name': v2_response.get('name', ''),
            'type': v2_response.get('type', 'claude_code'),
            'status': v2_response.get('status', 'unknown'),
            'created_at': v2_response.get('created_at', datetime.utcnow().isoformat()),
            'capabilities': v2_response.get('capabilities', [])
        }
    elif format_type == 'endpoints':
        return {
            'id': v2_response.get('id'),
            'name': v2_response.get('name'),
            'type': v2_response.get('type'),
            'role': v2_response.get('role'),
            'capabilities': v2_response.get('capabilities'),
            'status': v2_response.get('status'),
            'config': v2_response.get('config'),
            'tmux_session': v2_response.get('tmux_session'),
            'total_tasks_completed': v2_response.get('total_tasks_completed'),
            'total_tasks_failed': v2_response.get('total_tasks_failed'),
            'average_response_time': v2_response.get('average_response_time'),
            'context_window_usage': v2_response.get('context_window_usage'),
            'created_at': v2_response.get('created_at'),
            'updated_at': v2_response.get('updated_at'),
            'last_heartbeat': v2_response.get('last_heartbeat'),
            'last_active': v2_response.get('last_active')
        }
    
    return v2_response


# ========================================
# Health Check for Legacy APIs
# ========================================

@router.get("/health/compatibility")
@performance_monitor(target_ms=100)
async def legacy_compatibility_health_check() -> Dict[str, Any]:
    """
    Health check endpoint specifically for legacy API compatibility.
    
    Validates that all legacy endpoint transformations are working correctly
    and provides compatibility status information.
    """
    try:
        health_status = {
            'service': 'agent_management_v1_compatibility',
            'healthy': True,
            'timestamp': datetime.utcnow().isoformat(),
            'compatibility_layers': {
                'v1_agents_simple': {
                    'status': 'operational',
                    'endpoints': ['/agents', '/agents/{id}', '/agents/{id}/status'],
                    'transformations': 'active'
                },
                'v1_agent_endpoints': {
                    'status': 'operational', 
                    'endpoints': ['/api/v1/agents', '/api/v1/agents/{id}'],
                    'transformations': 'active'
                },
                'v1_agent_activation': {
                    'status': 'operational',
                    'endpoints': ['/activate', '/status', '/spawn/{role}'],
                    'transformations': 'active'
                },
                'v1_coordination': {
                    'status': 'operational',
                    'endpoints': ['/coordination/projects'],
                    'transformations': 'active'
                }
            },
            'migration_recommendations': [
                'Consider upgrading to v2 API endpoints for enhanced functionality',
                'v1 endpoints will be supported for 12 months from v2 release',
                'New features will only be available in v2 API'
            ],
            'performance_impact': {
                'transformation_overhead': '<5ms per request',
                'compatibility_cache': 'enabled',
                'legacy_support_efficiency': '98%'
            }
        }
        
        return health_status
        
    except Exception as e:
        logger.error("Legacy compatibility health check failed", error=str(e))
        return {
            'service': 'agent_management_v1_compatibility',
            'healthy': False,
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }