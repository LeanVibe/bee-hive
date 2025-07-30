"""
Enhanced Pydantic schemas for Team Coordination API.

Comprehensive request/response models demonstrating enterprise-grade validation,
data transformation, and API contract definition for multi-agent coordination.
"""

import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union, Literal
from enum import Enum

from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
from pydantic.types import PositiveInt, PositiveFloat, constr

from ..models.agent import AgentStatus, AgentType
from ..models.task import TaskStatus, TaskPriority, TaskType


# =====================================================================================
# ENUMS AND CONSTANTS
# =====================================================================================

class CoordinationMode(str, Enum):
    """Coordination modes for multi-agent workflows."""
    PARALLEL = "parallel"
    SEQUENTIAL = "sequential"
    HYBRID = "hybrid"
    AUTONOMOUS = "autonomous"


class CapabilityLevel(str, Enum):
    """Standardized capability proficiency levels."""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"
    MASTER = "master"


class MetricGranularity(str, Enum):
    """Time granularity for metrics aggregation."""
    MINUTE = "minute"
    HOUR = "hour" 
    DAY = "day"
    WEEK = "week"
    MONTH = "month"


class AssignmentStrategy(str, Enum):
    """Strategies for task assignment."""
    OPTIMAL_MATCH = "optimal_match"
    LOAD_BALANCED = "load_balanced"
    RANDOM = "random"
    ROUND_ROBIN = "round_robin"
    CAPABILITY_FOCUSED = "capability_focused"


# =====================================================================================
# BASE SCHEMAS AND MIXINS
# =====================================================================================

class TimestampMixin(BaseModel):
    """Mixin for consistent timestamp handling."""
    created_at: Optional[datetime] = Field(None, description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")


class PaginationMixin(BaseModel):
    """Mixin for pagination parameters."""
    offset: int = Field(default=0, ge=0, description="Number of items to skip")
    limit: int = Field(default=50, ge=1, le=500, description="Maximum items to return")


class MetricsMixin(BaseModel):
    """Mixin for performance metrics."""
    response_time_ms: Optional[float] = Field(None, ge=0, description="Response time in milliseconds")
    success_rate: Optional[float] = Field(None, ge=0.0, le=1.0, description="Success rate (0.0-1.0)")
    error_count: int = Field(default=0, ge=0, description="Number of errors")


# =====================================================================================
# CAPABILITY AND AGENT SCHEMAS
# =====================================================================================

class CapabilityDefinition(BaseModel):
    """Detailed capability definition with validation."""
    name: constr(min_length=1, max_length=100, strip_whitespace=True) = Field(
        ..., description="Capability name (unique identifier)"
    )
    description: constr(min_length=10, max_length=1000, strip_whitespace=True) = Field(
        ..., description="Detailed capability description"
    )
    category: constr(min_length=1, max_length=50) = Field(
        ..., description="Capability category (e.g., 'backend', 'frontend', 'devops')"
    )
    level: CapabilityLevel = Field(..., description="Proficiency level")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Self-assessed confidence (0.0-1.0)")
    
    # Experience and validation
    years_experience: Optional[PositiveFloat] = Field(None, description="Years of experience")
    certifications: List[str] = Field(default_factory=list, description="Relevant certifications")
    technologies: List[str] = Field(default_factory=list, description="Specific technologies/tools")
    
    # Performance indicators
    avg_task_completion_hours: Optional[PositiveFloat] = Field(None, description="Average task completion time")
    success_rate: Optional[float] = Field(None, ge=0.0, le=1.0, description="Historical success rate")
    
    @field_validator('technologies')
    @classmethod
    def validate_technologies(cls, v):
        """Validate and clean technology list."""
        if len(v) > 20:
            raise ValueError("Maximum 20 technologies allowed per capability")
        return [tech.strip() for tech in v if tech.strip()]
    
    @field_validator('certifications')
    @classmethod
    def validate_certifications(cls, v):
        """Validate certifications list."""
        if len(v) > 10:
            raise ValueError("Maximum 10 certifications allowed per capability")
        return [cert.strip() for cert in v if cert.strip()]


class WorkloadPreferences(BaseModel):
    """Agent workload and scheduling preferences."""
    max_concurrent_tasks: PositiveInt = Field(default=3, le=20, description="Maximum concurrent tasks")
    preferred_task_types: List[TaskType] = Field(default_factory=list, description="Preferred task types")
    avoided_task_types: List[TaskType] = Field(default_factory=list, description="Task types to avoid")
    
    # Time preferences
    working_hours_start: Optional[int] = Field(None, ge=0, le=23, description="Preferred start hour (0-23)")
    working_hours_end: Optional[int] = Field(None, ge=0, le=23, description="Preferred end hour (0-23)")
    timezone: Optional[str] = Field(None, description="Agent timezone (e.g., 'America/New_York')")
    
    # Break preferences
    break_duration_minutes: int = Field(default=15, ge=5, le=120, description="Preferred break duration")
    break_frequency_hours: float = Field(default=4.0, ge=1.0, le=12.0, description="Hours between breaks")
    
    @field_validator('preferred_task_types', 'avoided_task_types')
    @classmethod
    def validate_task_type_lists(cls, v):
        """Ensure task type lists don't exceed reasonable limits."""
        if len(v) > len(TaskType):
            raise ValueError("Cannot specify more task types than exist")
        return v
    
    @model_validator(mode='after')
    def validate_working_hours(self):
        """Validate working hours consistency."""
        start = self.working_hours_start
        end = self.working_hours_end
        
        if start is not None and end is not None:
            if start >= end:
                raise ValueError("Working hours start must be before end")
        
        return self


class AgentRegistrationRequest(BaseModel):
    """Comprehensive agent registration with enterprise validation."""
    # Basic information
    agent_name: constr(min_length=1, max_length=255, strip_whitespace=True) = Field(
        ..., description="Human-readable agent name"
    )
    agent_type: AgentType = Field(default=AgentType.CLAUDE, description="Agent implementation type")
    description: Optional[constr(max_length=1000)] = Field(None, description="Agent description")
    
    # Capabilities and skills
    capabilities: List[CapabilityDefinition] = Field(
        ..., min_items=1, max_items=20, description="Agent capabilities"
    )
    primary_role: constr(min_length=1, max_length=100) = Field(
        ..., description="Primary role (e.g., 'Backend Developer', 'QA Engineer')"
    )
    secondary_roles: List[str] = Field(default_factory=list, description="Secondary roles")
    
    # System configuration
    system_prompt: Optional[constr(max_length=5000)] = Field(
        None, description="System prompt/context for AI agents"
    )
    config_parameters: Dict[str, Any] = Field(
        default_factory=dict, description="Agent-specific configuration"
    )
    
    # Workload and preferences
    workload_preferences: WorkloadPreferences = Field(
        default_factory=WorkloadPreferences, description="Workload and scheduling preferences"
    )
    
    # Metadata and classification
    tags: List[str] = Field(default_factory=list, description="Classification tags")
    team_assignments: List[str] = Field(default_factory=list, description="Team memberships")
    reporting_manager: Optional[str] = Field(None, description="Manager/supervisor agent ID")
    
    # Contact and integration
    notification_channels: List[str] = Field(default_factory=list, description="Notification endpoints")
    integration_webhooks: Dict[str, str] = Field(
        default_factory=dict, description="Webhook URLs for integrations"
    )
    
    @field_validator('capabilities')
    @classmethod
    def validate_unique_capabilities(cls, v):
        """Ensure capability names are unique."""
        names = [cap.name.lower() for cap in v]
        if len(names) != len(set(names)):
            raise ValueError("Capability names must be unique")
        return v
    
    @field_validator('tags')
    @classmethod
    def validate_tags(cls, v):
        """Validate and clean tags."""
        if len(v) > 20:
            raise ValueError("Maximum 20 tags allowed")
        return [tag.strip().lower() for tag in v if tag.strip()]
    
    @field_validator('secondary_roles')
    @classmethod
    def validate_secondary_roles(cls, v):
        """Validate secondary roles."""
        if len(v) > 5:
            raise ValueError("Maximum 5 secondary roles allowed")
        return [role.strip() for role in v if role.strip()]


# =====================================================================================
# TASK DISTRIBUTION SCHEMAS
# =====================================================================================

class TaskRequirements(BaseModel):
    """Detailed task requirements for intelligent distribution."""
    required_capabilities: List[str] = Field(..., min_items=1, description="Required capabilities")
    preferred_capabilities: List[str] = Field(default_factory=list, description="Nice-to-have capabilities")
    
    # Complexity and effort
    complexity_score: float = Field(default=0.5, ge=0.0, le=1.0, description="Task complexity (0.0-1.0)")
    estimated_effort_hours: Optional[PositiveFloat] = Field(None, le=200, description="Estimated effort")
    urgency_multiplier: float = Field(default=1.0, ge=0.1, le=5.0, description="Urgency multiplier")
    
    # Technical requirements
    technologies_required: List[str] = Field(default_factory=list, description="Required technologies")
    minimum_experience_years: Optional[PositiveFloat] = Field(None, description="Minimum experience required")
    certification_required: Optional[str] = Field(None, description="Required certification")
    
    # Context requirements
    domain_knowledge: List[str] = Field(default_factory=list, description="Required domain knowledge")
    previous_project_experience: List[str] = Field(default_factory=list, description="Relevant project types")
    
    @field_validator('required_capabilities', 'preferred_capabilities')
    @classmethod
    def validate_capability_lists(cls, v):
        """Validate capability requirement lists."""
        if len(v) > 15:
            raise ValueError("Maximum 15 capabilities per list")
        return [cap.strip() for cap in v if cap.strip()]


class TaskConstraints(BaseModel):
    """Task execution constraints and preferences."""
    # Time constraints
    deadline: Optional[datetime] = Field(None, description="Hard deadline")
    preferred_completion_date: Optional[datetime] = Field(None, description="Preferred completion date")
    max_duration_days: Optional[PositiveInt] = Field(None, description="Maximum duration in days")
    
    # Agent constraints
    excluded_agents: List[str] = Field(default_factory=list, description="Agents to exclude")
    preferred_agents: List[str] = Field(default_factory=list, description="Preferred agents")
    require_human_approval: bool = Field(default=False, description="Requires human approval")
    
    # Resource constraints
    max_resource_usage: Optional[float] = Field(None, ge=0.0, le=1.0, description="Max resource usage")
    requires_gpu: bool = Field(default=False, description="Requires GPU resources")
    memory_requirements_gb: Optional[PositiveFloat] = Field(None, description="Memory requirements")
    
    # Collaboration constraints
    allow_collaboration: bool = Field(default=True, description="Allow multiple agents")
    max_collaborators: int = Field(default=3, ge=1, le=10, description="Maximum collaborating agents")
    requires_code_review: bool = Field(default=True, description="Requires code review")
    
    @model_validator(mode='after')
    def validate_date_constraints(self):
        """Validate date constraint consistency."""
        deadline = self.deadline
        preferred = self.preferred_completion_date
        
        if deadline and preferred and preferred > deadline:
            raise ValueError("Preferred completion date cannot be after deadline")
        
        return self


class TaskDistributionRequest(BaseModel):
    """Comprehensive task distribution request."""
    # Basic task information
    title: constr(min_length=1, max_length=255, strip_whitespace=True) = Field(
        ..., description="Task title"
    )
    description: constr(min_length=10, strip_whitespace=True) = Field(
        ..., description="Detailed task description"
    )
    task_type: TaskType = Field(..., description="Type of task")
    priority: TaskPriority = Field(default=TaskPriority.MEDIUM, description="Task priority")
    
    # Requirements and constraints
    requirements: TaskRequirements = Field(..., description="Task requirements")
    constraints: TaskConstraints = Field(default_factory=TaskConstraints, description="Task constraints")
    
    # Assignment preferences
    assignment_strategy: AssignmentStrategy = Field(
        default=AssignmentStrategy.OPTIMAL_MATCH, description="Assignment strategy"
    )
    allow_auto_assignment: bool = Field(default=True, description="Allow automatic assignment")
    
    # Context and dependencies
    project_context: Optional[str] = Field(None, description="Project context information")
    dependencies: List[str] = Field(default_factory=list, description="Dependent task IDs")
    blocks_tasks: List[str] = Field(default_factory=list, description="Tasks this blocks")
    
    # Metadata
    labels: List[str] = Field(default_factory=list, description="Task labels")
    custom_fields: Dict[str, Any] = Field(default_factory=dict, description="Custom fields")
    
    @field_validator('dependencies', 'blocks_tasks')
    @classmethod
    def validate_task_relationships(cls, v):
        """Validate task relationship lists."""
        if len(v) > 20:
            raise ValueError("Maximum 20 task relationships allowed")
        # Validate UUID format
        for task_id in v:
            try:
                uuid.UUID(task_id)
            except ValueError:
                raise ValueError(f"Invalid task ID format: {task_id}")
        return v


# =====================================================================================
# RESPONSE SCHEMAS
# =====================================================================================

class CapabilityMatchDetails(BaseModel):
    """Detailed capability matching information."""
    capability_name: str
    required_level: str
    agent_level: str
    match_score: float = Field(..., ge=0.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    gap_analysis: Optional[str] = None


class AgentSuitabilityScore(BaseModel):
    """Agent suitability analysis for task assignment."""
    agent_id: str
    agent_name: str
    overall_score: float = Field(..., ge=0.0, le=1.0, description="Overall suitability score")
    
    # Component scores
    capability_score: float = Field(..., ge=0.0, le=1.0, description="Capability match score")
    workload_score: float = Field(..., ge=0.0, le=1.0, description="Workload availability score")
    performance_score: float = Field(..., ge=0.0, le=1.0, description="Historical performance score")
    preference_score: float = Field(..., ge=0.0, le=1.0, description="Task preference alignment")
    
    # Detailed analysis
    capability_matches: List[CapabilityMatchDetails]
    current_workload: float = Field(..., ge=0.0, le=1.0)
    estimated_completion_time: Optional[datetime] = None
    risk_factors: List[str] = Field(default_factory=list)
    
    # Recommendation
    recommendation: Literal["strongly_recommended", "recommended", "suitable", "not_suitable"]
    reasoning: str


class TaskDistributionResponse(BaseModel):
    """Comprehensive task distribution response."""
    task_id: str
    status: Literal["assigned", "queued", "rejected"]
    
    # Assignment details
    assigned_agent_id: Optional[str] = None
    assigned_agent_name: Optional[str] = None
    assignment_timestamp: datetime
    estimated_start_time: Optional[datetime] = None
    estimated_completion_time: Optional[datetime] = None
    
    # Assignment analysis
    assignment_confidence: float = Field(..., ge=0.0, le=1.0)
    suitability_analysis: Optional[AgentSuitabilityScore] = None
    alternative_agents: List[AgentSuitabilityScore] = Field(default_factory=list)
    
    # Impact analysis
    workload_impact: Dict[str, float] = Field(default_factory=dict)
    resource_allocation: Dict[str, Any] = Field(default_factory=dict)
    
    # Metadata
    assignment_strategy_used: AssignmentStrategy
    rejection_reason: Optional[str] = None
    queue_position: Optional[int] = None


class AgentPerformanceMetrics(BaseModel):
    """Comprehensive agent performance metrics."""
    agent_id: str
    agent_name: str
    measurement_period: str
    
    # Task completion metrics
    tasks_assigned: int = Field(..., ge=0)
    tasks_completed: int = Field(..., ge=0)
    tasks_failed: int = Field(..., ge=0)
    completion_rate: float = Field(..., ge=0.0, le=1.0)
    
    # Time metrics
    avg_response_time_hours: float = Field(..., ge=0.0)
    avg_completion_time_hours: float = Field(..., ge=0.0)
    on_time_delivery_rate: float = Field(..., ge=0.0, le=1.0)
    
    # Quality metrics
    code_review_pass_rate: float = Field(..., ge=0.0, le=1.0)
    bug_introduction_rate: float = Field(..., ge=0.0)
    customer_satisfaction_score: Optional[float] = Field(None, ge=0.0, le=5.0)
    
    # Utilization metrics
    utilization_rate: float = Field(..., ge=0.0, le=1.0)
    idle_time_percentage: float = Field(..., ge=0.0, le=1.0)
    overtime_hours: float = Field(..., ge=0.0)
    
    # Capability development
    new_capabilities_acquired: int = Field(..., ge=0)
    capability_improvement_score: float = Field(..., ge=0.0, le=1.0)
    learning_velocity: float = Field(..., ge=0.0)


class SystemCoordinationMetrics(BaseModel):
    """System-wide coordination metrics."""
    timestamp: datetime
    measurement_window_hours: int
    
    # Agent metrics
    total_agents: int = Field(..., ge=0)
    active_agents: int = Field(..., ge=0)
    idle_agents: int = Field(..., ge=0)
    overloaded_agents: int = Field(..., ge=0)
    
    # Task metrics
    total_tasks: int = Field(..., ge=0)
    pending_tasks: int = Field(..., ge=0)
    active_tasks: int = Field(..., ge=0)
    completed_tasks: int = Field(..., ge=0)
    failed_tasks: int = Field(..., ge=0)
    
    # System efficiency
    overall_utilization_rate: float = Field(..., ge=0.0, le=1.0)
    task_assignment_success_rate: float = Field(..., ge=0.0, le=1.0)
    average_queue_time_minutes: float = Field(..., ge=0.0)
    system_throughput_tasks_per_hour: float = Field(..., ge=0.0)
    
    # Quality indicators
    deadline_adherence_rate: float = Field(..., ge=0.0, le=1.0)
    rework_percentage: float = Field(..., ge=0.0, le=1.0)
    escalation_rate: float = Field(..., ge=0.0, le=1.0)
    
    # Bottleneck analysis
    bottleneck_capabilities: List[str] = Field(default_factory=list)
    oversubscribed_skills: List[str] = Field(default_factory=list)
    underutilized_skills: List[str] = Field(default_factory=list)
    
    # Recommendations
    scaling_recommendations: List[str] = Field(default_factory=list)
    optimization_opportunities: List[str] = Field(default_factory=list)


# =====================================================================================
# WEBSOCKET AND REAL-TIME SCHEMAS
# =====================================================================================

class WebSocketMessage(BaseModel):
    """Base WebSocket message structure."""
    type: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    correlation_id: Optional[str] = None


class AgentStatusUpdate(WebSocketMessage):
    """Agent status update message."""
    type: Literal["agent_status_update"] = "agent_status_update"
    agent_id: str
    agent_name: str
    old_status: AgentStatus
    new_status: AgentStatus
    current_workload: float
    active_tasks: int
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TaskStatusUpdate(WebSocketMessage):
    """Task status update message."""
    type: Literal["task_status_update"] = "task_status_update"
    task_id: str
    task_title: str
    old_status: TaskStatus
    new_status: TaskStatus
    assigned_agent_id: Optional[str] = None
    assigned_agent_name: Optional[str] = None
    completion_percentage: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SystemMetricsUpdate(WebSocketMessage):
    """System metrics update message."""
    type: Literal["system_metrics_update"] = "system_metrics_update"
    metrics: Dict[str, Union[int, float, str]]
    alerts: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)


class CoordinationAlert(WebSocketMessage):
    """Coordination system alert message."""
    type: Literal["coordination_alert"] = "coordination_alert"
    severity: Literal["info", "warning", "error", "critical"]
    alert_type: str
    message: str
    affected_components: List[str] = Field(default_factory=list)
    recommended_actions: List[str] = Field(default_factory=list)
    auto_resolution_attempted: bool = False


# =====================================================================================
# QUERY AND FILTER SCHEMAS
# =====================================================================================

class AgentFilterQuery(PaginationMixin, BaseModel):
    """Advanced agent filtering query."""
    # Status filters
    status: Optional[List[AgentStatus]] = Field(None, description="Filter by agent status")
    agent_type: Optional[List[AgentType]] = Field(None, description="Filter by agent type")
    
    # Capability filters
    has_capability: Optional[List[str]] = Field(None, description="Must have these capabilities")
    capability_level: Optional[CapabilityLevel] = Field(None, description="Minimum capability level")
    
    # Workload filters
    max_workload: Optional[float] = Field(None, ge=0.0, le=1.0, description="Maximum current workload")
    min_availability: Optional[float] = Field(None, ge=0.0, le=1.0, description="Minimum availability")
    
    # Performance filters
    min_completion_rate: Optional[float] = Field(None, ge=0.0, le=1.0, description="Minimum completion rate")
    max_response_time: Optional[float] = Field(None, ge=0.0, description="Maximum response time")
    
    # Metadata filters
    tags: Optional[List[str]] = Field(None, description="Must have these tags")
    team: Optional[str] = Field(None, description="Team assignment")
    
    # Sorting
    sort_by: Optional[str] = Field("name", description="Sort field")
    sort_order: Literal["asc", "desc"] = Field("asc", description="Sort order")


class TaskFilterQuery(PaginationMixin, BaseModel):
    """Advanced task filtering query."""
    # Status filters
    status: Optional[List[TaskStatus]] = Field(None, description="Filter by task status")
    task_type: Optional[List[TaskType]] = Field(None, description="Filter by task type")
    priority: Optional[List[TaskPriority]] = Field(None, description="Filter by priority")
    
    # Assignment filters
    assigned_agent_id: Optional[str] = Field(None, description="Filter by assigned agent")
    unassigned_only: bool = Field(False, description="Show only unassigned tasks")
    
    # Time filters
    created_after: Optional[datetime] = Field(None, description="Created after this date")
    created_before: Optional[datetime] = Field(None, description="Created before this date")
    due_before: Optional[datetime] = Field(None, description="Due before this date")
    
    # Capability filters
    requires_capability: Optional[List[str]] = Field(None, description="Requires these capabilities")
    
    # Text search
    search_text: Optional[str] = Field(None, description="Search in title and description")
    
    # Sorting
    sort_by: Optional[str] = Field("created_at", description="Sort field")
    sort_order: Literal["asc", "desc"] = Field("desc", description="Sort order")


class MetricsQuery(BaseModel):
    """Metrics query parameters."""
    # Time range
    start_time: Optional[datetime] = Field(None, description="Start of time range")
    end_time: Optional[datetime] = Field(None, description="End of time range")
    time_range_hours: Optional[int] = Field(None, ge=1, le=8760, description="Time range in hours")
    
    # Granularity
    granularity: MetricGranularity = Field(MetricGranularity.HOUR, description="Aggregation granularity")
    
    # Filters
    agent_ids: Optional[List[str]] = Field(None, description="Specific agents to include")
    task_types: Optional[List[TaskType]] = Field(None, description="Task types to include")
    
    # Metric selection
    include_performance: bool = Field(True, description="Include performance metrics")
    include_utilization: bool = Field(True, description="Include utilization metrics")
    include_quality: bool = Field(True, description="Include quality metrics")
    
    @model_validator(mode='after')
    def validate_time_range(self):
        """Validate time range parameters."""
        start_time = self.start_time
        end_time = self.end_time
        time_range_hours = self.time_range_hours
        
        if time_range_hours:
            if start_time or end_time:
                raise ValueError("Cannot specify both time_range_hours and start_time/end_time")
        elif not (start_time and end_time):
            # Default to last 24 hours
            self.time_range_hours = 24
        elif start_time and end_time and start_time >= end_time:
            raise ValueError("start_time must be before end_time")
            
        return self


# =====================================================================================
# ERROR AND VALIDATION SCHEMAS
# =====================================================================================

class ValidationError(BaseModel):
    """Detailed validation error information."""
    field: str
    message: str
    invalid_value: Any
    constraint: Optional[str] = None


class APIError(BaseModel):
    """Standardized API error response."""
    error_code: str
    message: str
    details: Optional[str] = None
    validation_errors: List[ValidationError] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    request_id: Optional[str] = None
    help_url: Optional[str] = None


class HealthCheckResponse(BaseModel):
    """System health check response."""
    status: Literal["healthy", "degraded", "unhealthy"]
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    version: str
    uptime_seconds: float
    
    components: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    metrics: Dict[str, Union[int, float]] = Field(default_factory=dict)
    alerts: List[str] = Field(default_factory=list)


# Model configuration for all schemas
for schema_class in [
    AgentRegistrationRequest, TaskDistributionRequest, TaskDistributionResponse,
    AgentPerformanceMetrics, SystemCoordinationMetrics, WebSocketMessage,
    AgentStatusUpdate, TaskStatusUpdate, SystemMetricsUpdate, CoordinationAlert
]:
    schema_class.model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        use_enum_values=True,
        json_encoders={
            datetime: lambda v: v.isoformat() + "Z"
        }
    )