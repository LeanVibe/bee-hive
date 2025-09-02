"""
Unified Data Models for AgentManagementAPI v2

Consolidates all agent management data models from 6 source modules into a single,
consistent data model framework following OpenAPI 3.0 specifications.
"""

import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from enum import Enum

from pydantic import BaseModel, Field, validator
try:
    from ....models.agent import AgentStatus, AgentType
except ImportError:
    # Fallback imports if models are in different location
    from enum import Enum
    
    class AgentStatus(Enum):
        CREATED = "CREATED"
        ACTIVE = "ACTIVE"
        INACTIVE = "INACTIVE"
        IDLE = "IDLE"
        BUSY = "BUSY"
        ERROR = "ERROR"
        FAILED = "FAILED"
        UNKNOWN = "UNKNOWN"
    
    class AgentType(Enum):
        CLAUDE = "claude"
        OPENAI = "openai"
        CLAUDE_CODE = "claude_code"

try:
    from ....core.coordination import CoordinationMode, ProjectStatus
except ImportError:
    # Fallback enums if coordination module not available
    from enum import Enum
    
    class CoordinationMode(Enum):
        PARALLEL = "PARALLEL"
        SEQUENTIAL = "SEQUENTIAL"
        HYBRID = "HYBRID"
        ADAPTIVE = "ADAPTIVE"
    
    class ProjectStatus(Enum):
        CREATED = "CREATED"
        ACTIVE = "ACTIVE"
        COMPLETED = "COMPLETED"
        FAILED = "FAILED"
        CANCELLED = "CANCELLED"


# ========================================
# Core Agent Management Models
# ========================================

class AgentCreateRequest(BaseModel):
    """Unified request model for agent creation across all API versions."""
    name: str = Field(..., min_length=1, max_length=100, description="Agent name")
    role: str = Field(..., description="Agent role (backend_developer, frontend_developer, etc.)")
    type: str = Field(default="claude_code", description="Type of agent to create")
    capabilities: List[str] = Field(default_factory=list, description="Agent capabilities")
    system_prompt: Optional[str] = Field(None, description="Custom system prompt for agent")
    config: Optional[Dict[str, Any]] = Field(None, description="Agent configuration parameters")
    
    # Coordination-specific fields
    task_id: Optional[str] = Field(None, description="Optional task ID to assign")
    workspace_name: Optional[str] = Field(None, description="Workspace name")
    git_branch: Optional[str] = Field(None, description="Git branch for workspace")
    
    @validator('role')
    def validate_role(cls, v):
        valid_roles = ['backend_developer', 'frontend_developer', 'devops_engineer', 'qa_engineer', 'general']
        if v.lower() not in valid_roles:
            raise ValueError(f'Role must be one of: {valid_roles}')
        return v.lower()


class AgentResponse(BaseModel):
    """Unified response model for agent data across all API versions."""
    id: Union[str, uuid.UUID]
    name: str
    role: str
    type: str = "claude_code"
    status: str
    capabilities: List[str] = Field(default_factory=list)
    config: Dict[str, Any] = Field(default_factory=dict)
    
    # Performance and monitoring fields
    tmux_session: Optional[str] = None
    total_tasks_completed: int = 0
    total_tasks_failed: int = 0
    average_response_time: float = 0.0
    context_window_usage: float = 0.0
    
    # Timestamp fields
    created_at: Union[str, datetime]
    updated_at: Optional[Union[str, datetime]] = None
    last_heartbeat: Optional[Union[str, datetime]] = None
    last_active: Optional[Union[str, datetime]] = None
    
    # Coordination fields
    current_task_id: Optional[str] = None
    assigned_projects: List[str] = Field(default_factory=list)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            uuid.UUID: lambda v: str(v)
        }


class AgentListResponse(BaseModel):
    """Response model for agent list operations."""
    agents: List[AgentResponse]
    total: int
    active: int = 0
    inactive: int = 0
    offset: int = 0
    limit: int = 50


class AgentStatsResponse(BaseModel):
    """Response model for detailed agent statistics."""
    agent_id: Union[str, uuid.UUID]
    total_tasks_completed: int = 0
    total_tasks_failed: int = 0
    success_rate: float = 0.0
    average_response_time: float = 0.0
    context_window_usage: float = 0.0
    uptime_hours: float = 0.0
    last_active: Optional[Union[str, datetime]] = None
    capabilities_count: int = 0
    performance_trend: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            uuid.UUID: lambda v: str(v)
        }


class AgentStatusUpdateRequest(BaseModel):
    """Request model for agent status updates."""
    status: str = Field(..., description="New agent status")
    reason: Optional[str] = Field(None, description="Reason for status change")
    
    @validator('status')
    def validate_status(cls, v):
        valid_statuses = [s.value.upper() for s in AgentStatus]
        if v.upper() not in valid_statuses:
            raise ValueError(f'Status must be one of: {valid_statuses}')
        return v.upper()


class AgentOperationResponse(BaseModel):
    """Standard response model for agent operations."""
    success: bool = True
    message: str
    agent_id: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    operation_details: Optional[Dict[str, Any]] = None


# ========================================
# Agent Coordination Models
# ========================================

class ProjectCreateRequest(BaseModel):
    """Request to create a coordinated multi-agent project."""
    name: str = Field(..., description="Project name")
    description: str = Field(..., description="Project description")
    requirements: Dict[str, Any] = Field(..., description="Project requirements")
    coordination_mode: str = Field(default="parallel", description="Coordination mode")
    deadline: Optional[str] = Field(None, description="Project deadline (ISO format)")
    quality_gates: Optional[List[Dict[str, Any]]] = Field(None, description="Custom quality gates")
    preferred_agents: Optional[List[str]] = Field(None, description="Preferred agent IDs")
    
    @validator('coordination_mode')
    def validate_coordination_mode(cls, v):
        valid_modes = ['parallel', 'sequential', 'hybrid', 'adaptive']
        if v.lower() not in valid_modes:
            raise ValueError(f'Coordination mode must be one of: {valid_modes}')
        return v.lower()


class ProjectResponse(BaseModel):
    """Response with project information."""
    project_id: str
    name: str
    description: str
    status: str
    coordination_mode: str
    participating_agents: List[str] = Field(default_factory=list)
    progress_percentage: float = 0.0
    created_at: str
    started_at: Optional[str] = None
    estimated_completion: Optional[str] = None
    quality_gates_passed: int = 0
    active_conflicts: int = 0


class ProjectStatusResponse(BaseModel):
    """Detailed project status response."""
    project_id: str
    name: str
    status: str
    progress_metrics: Dict[str, Any] = Field(default_factory=dict)
    quality_gates: List[Dict[str, Any]] = Field(default_factory=list)
    participating_agents: List[str] = Field(default_factory=list)
    active_conflicts: List[str] = Field(default_factory=list)
    tasks_summary: Dict[str, int] = Field(default_factory=dict)
    agent_utilization: float = 0.0
    performance_insights: Dict[str, Any] = Field(default_factory=dict)


class AgentRegistrationRequest(BaseModel):
    """Request to register an agent for coordination."""
    agent_id: str = Field(..., description="Agent ID")
    capabilities: List[str] = Field(..., description="Agent capabilities")
    specializations: List[str] = Field(..., description="Agent specializations")
    proficiency: float = Field(default=0.8, ge=0.0, le=1.0, description="Proficiency level")
    experience_level: str = Field(default="intermediate", description="Experience level")
    availability: Dict[str, Any] = Field(default_factory=dict, description="Agent availability schedule")


class TaskReassignmentRequest(BaseModel):
    """Request to reassign a task to different agent."""
    project_id: str = Field(..., description="Project ID")
    task_id: str = Field(..., description="Task ID")
    new_agent_id: str = Field(..., description="New agent ID")
    reason: str = Field(..., description="Reason for reassignment")
    priority: str = Field(default="normal", description="Reassignment priority")


class ConflictResponse(BaseModel):
    """Response with conflict information."""
    conflict_id: str
    project_id: str
    conflict_type: str
    severity: str
    description: str
    affected_agents: List[str] = Field(default_factory=list)
    resolution_status: str
    detected_at: str
    impact_score: float = 0.0
    suggested_resolution: Optional[str] = None


class ConflictResolutionRequest(BaseModel):
    """Request to manually resolve a conflict."""
    conflict_id: str = Field(..., description="Conflict ID")
    resolution_strategy: str = Field(..., description="Resolution strategy")
    resolution_data: Optional[Dict[str, Any]] = Field(None, description="Additional resolution data")
    notify_agents: bool = Field(default=True, description="Notify affected agents")


# ========================================
# Agent Activation Models
# ========================================

class AgentActivationRequest(BaseModel):
    """Request to activate the multi-agent system."""
    team_size: int = Field(default=5, ge=1, le=20, description="Size of agent team to activate")
    roles: Optional[List[str]] = Field(None, description="Specific roles to activate")
    auto_start_tasks: bool = Field(default=True, description="Automatically start demo tasks")
    workspace_config: Optional[Dict[str, Any]] = Field(None, description="Workspace configuration")


class AgentActivationResponse(BaseModel):
    """Response from agent activation."""
    success: bool
    message: str
    active_agents: Dict[str, Any] = Field(default_factory=dict)
    team_composition: Dict[str, str] = Field(default_factory=dict)
    activation_time: float = 0.0
    system_status: str = "ready"


class SystemStatusResponse(BaseModel):
    """System status response for agent management."""
    active: bool
    agent_count: int
    system_ready: bool
    orchestrator_type: str = "ConsolidatedProductionOrchestrator"
    orchestrator_health: str = "healthy"
    performance: Dict[str, Any] = Field(default_factory=dict)
    agents: Dict[str, Any] = Field(default_factory=dict)
    capabilities_summary: List[str] = Field(default_factory=list)
    last_update: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


# ========================================
# WebSocket and Real-time Models
# ========================================

class WebSocketMessage(BaseModel):
    """WebSocket message model for real-time coordination updates."""
    type: str = Field(..., description="Message type")
    data: Dict[str, Any] = Field(default_factory=dict, description="Message data")
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    source: str = Field(default="agent_management", description="Message source")


class CoordinationUpdate(BaseModel):
    """Real-time coordination update message."""
    project_id: str
    update_type: str  # "status_change", "task_completed", "conflict_detected", etc.
    affected_agents: List[str] = Field(default_factory=list)
    data: Dict[str, Any] = Field(default_factory=dict)
    priority: str = Field(default="normal", description="Update priority")


# ========================================
# Health and Monitoring Models
# ========================================

class AgentHealthResponse(BaseModel):
    """Agent health status response."""
    agent_id: str
    status: str
    last_activity: str
    uptime_seconds: float
    current_task: Optional[str] = None
    healthy: bool = True
    session: Optional[Dict[str, Any]] = None
    performance_metrics: Dict[str, Any] = Field(default_factory=dict)
    resource_usage: Dict[str, Any] = Field(default_factory=dict)


class CoordinationMetricsResponse(BaseModel):
    """Coordination engine performance metrics."""
    coordination_metrics: Dict[str, Any] = Field(default_factory=dict)
    real_time_stats: Dict[str, Any] = Field(default_factory=dict)
    performance_indicators: Dict[str, Any] = Field(default_factory=dict)
    agent_utilization_trends: List[Dict[str, Any]] = Field(default_factory=list)


class AgentCapabilitiesResponse(BaseModel):
    """Agent capabilities summary."""
    total_agents: int
    roles: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    system_capabilities: List[str] = Field(default_factory=list)
    capability_matrix: Dict[str, List[str]] = Field(default_factory=dict)
    specialization_coverage: float = 0.0


# ========================================
# Compatibility Models for v1 API
# ========================================

class LegacyAgentResponse(BaseModel):
    """Legacy response format for v1 compatibility."""
    id: str
    name: str
    type: str
    status: str
    created_at: str
    capabilities: List[str] = Field(default_factory=list)
    
    @classmethod
    def from_unified_response(cls, unified: AgentResponse) -> 'LegacyAgentResponse':
        """Convert unified response to legacy format."""
        return cls(
            id=str(unified.id),
            name=unified.name,
            type=unified.type,
            status=unified.status,
            created_at=str(unified.created_at),
            capabilities=unified.capabilities
        )


# ========================================
# Validation and Error Models
# ========================================

class ValidationError(BaseModel):
    """Validation error response model."""
    field: str
    message: str
    code: str = "validation_error"


class ErrorResponse(BaseModel):
    """Standard error response model."""
    success: bool = False
    error: str
    details: Optional[List[ValidationError]] = None
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    trace_id: Optional[str] = None