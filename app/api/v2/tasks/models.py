"""
TaskExecutionAPI Models - Unified Data Models and Schemas

Consolidated Pydantic models for task execution, workflow orchestration,
and intelligent scheduling following Phase 2-3 architectural patterns.

Provides unified schema definitions combining patterns from:
- Task management endpoints
- Workflow orchestration
- Intelligent scheduling
- Team coordination
- Orchestrator core functionality
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from uuid import UUID
from enum import Enum

from pydantic import BaseModel, Field, validator, ConfigDict
from fastapi import Query

from ....models.task import TaskStatus, TaskPriority, TaskType
from ....models.workflow import WorkflowStatus, WorkflowPriority


# ===============================================================================
# CORE TASK EXECUTION MODELS
# ===============================================================================

class TaskExecutionRequest(BaseModel):
    """Unified request model for task execution creation."""
    model_config = ConfigDict(from_attributes=True)
    
    title: str = Field(..., min_length=1, max_length=255, description="Task title")
    description: Optional[str] = Field(None, max_length=2000, description="Detailed task description")
    task_type: Optional[TaskType] = Field(None, description="Type of development task")
    priority: TaskPriority = Field(default=TaskPriority.MEDIUM, description="Task priority level")
    
    # Capability and assignment
    required_capabilities: List[str] = Field(default_factory=list, description="Required agent capabilities")
    preferred_agent_id: Optional[str] = Field(None, description="Preferred agent for assignment")
    team_coordination: bool = Field(default=False, description="Enable team coordination features")
    
    # Timing and effort
    estimated_effort: Optional[int] = Field(None, ge=1, description="Estimated effort in minutes")
    deadline: Optional[datetime] = Field(None, description="Task deadline")
    timeout_seconds: Optional[int] = Field(None, ge=1, le=86400, description="Task timeout")
    
    # Context and configuration
    context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Task execution context")
    dependencies: List[str] = Field(default_factory=list, description="Task dependency IDs")
    
    # Execution options
    auto_assign: bool = Field(default=True, description="Automatically assign to available agent")
    enable_intelligent_scheduling: bool = Field(default=True, description="Use intelligent scheduling")
    workflow_integration: bool = Field(default=False, description="Enable workflow integration")
    
    @validator('required_capabilities')
    def validate_capabilities(cls, v):
        if len(v) > 20:
            raise ValueError("Maximum 20 required capabilities allowed")
        return [cap.strip() for cap in v if cap.strip()]


class TaskExecutionResponse(BaseModel):
    """Unified response model for task execution operations."""
    model_config = ConfigDict(from_attributes=True)
    
    # Core task information
    task_id: str
    title: str
    description: Optional[str] = None
    task_type: Optional[str] = None
    status: str
    priority: str
    
    # Assignment and execution
    assigned_agent_id: Optional[str] = None
    assigned_agent_name: Optional[str] = None
    assignment_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    
    # Timing information
    created_at: datetime
    assigned_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    estimated_completion: Optional[datetime] = None
    
    # Execution details
    progress: float = Field(default=0.0, ge=0.0, le=1.0, description="Completion progress")
    estimated_effort: Optional[int] = None
    actual_effort: Optional[int] = None
    
    # Results and errors
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    retry_count: int = Field(default=0, ge=0)
    
    # Metadata
    context: Optional[Dict[str, Any]] = None
    performance_metrics: Optional[Dict[str, Any]] = None
    

class TaskStatusUpdateRequest(BaseModel):
    """Request model for task status updates."""
    status: TaskStatus = Field(..., description="New task status")
    progress: Optional[float] = Field(None, ge=0.0, le=1.0, description="Progress percentage")
    result: Optional[Dict[str, Any]] = Field(None, description="Task execution result")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    actual_effort: Optional[int] = Field(None, ge=0, description="Actual effort in minutes")
    

class TaskPriorityUpdateRequest(BaseModel):
    """Request model for task priority updates."""
    priority: TaskPriority = Field(..., description="New task priority")
    reason: Optional[str] = Field(None, max_length=500, description="Reason for priority change")
    force_update: bool = Field(default=False, description="Force update even if task is running")


class TaskAssignmentRequest(BaseModel):
    """Request model for task assignment operations."""
    agent_id: Optional[str] = Field(None, description="Specific agent ID for assignment")
    strategy: Optional[str] = Field("intelligent", description="Assignment strategy")
    priority_override: Optional[TaskPriority] = Field(None, description="Priority override")
    context_override: Optional[Dict[str, Any]] = Field(None, description="Context override")
    force_assignment: bool = Field(default=False, description="Force assignment to busy agent")
    timeout_seconds: Optional[float] = Field(None, ge=1.0, le=300.0, description="Assignment timeout")


# ===============================================================================
# WORKFLOW ORCHESTRATION MODELS  
# ===============================================================================

class WorkflowExecutionRequest(BaseModel):
    """Request model for workflow execution."""
    model_config = ConfigDict(from_attributes=True)
    
    name: str = Field(..., min_length=1, max_length=255, description="Workflow name")
    description: Optional[str] = Field(None, max_length=2000, description="Workflow description")
    priority: WorkflowPriority = Field(default=WorkflowPriority.MEDIUM, description="Workflow priority")
    
    # Workflow definition
    definition: Dict[str, Any] = Field(..., description="Workflow definition structure")
    task_ids: List[str] = Field(default_factory=list, description="Task IDs to include")
    dependencies: Optional[Dict[str, List[str]]] = Field(None, description="Task dependencies")
    
    # Execution configuration
    context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Workflow context")
    variables: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Workflow variables")
    estimated_duration: Optional[int] = Field(None, ge=1, description="Estimated duration in minutes")
    due_date: Optional[datetime] = Field(None, description="Workflow due date")
    
    # Orchestration options
    parallel_execution: bool = Field(default=True, description="Enable parallel task execution")
    intelligent_scheduling: bool = Field(default=True, description="Use intelligent task scheduling")
    team_coordination: bool = Field(default=True, description="Enable team coordination")
    auto_recovery: bool = Field(default=True, description="Enable automatic error recovery")


class WorkflowExecutionResponse(BaseModel):
    """Response model for workflow execution operations."""
    model_config = ConfigDict(from_attributes=True)
    
    # Core workflow information
    workflow_id: str
    name: str
    description: Optional[str] = None
    status: str
    priority: str
    
    # Execution progress
    completion_percentage: float = Field(ge=0.0, le=100.0)
    total_tasks: int = Field(ge=0)
    completed_tasks: int = Field(ge=0)
    failed_tasks: int = Field(ge=0)
    pending_tasks: int = Field(ge=0)
    
    # Timing information
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    estimated_completion: Optional[datetime] = None
    
    # Execution details
    current_tasks: List[str] = Field(default_factory=list, description="Currently executing tasks")
    ready_tasks: List[str] = Field(default_factory=list, description="Ready to execute tasks")
    blocked_tasks: List[str] = Field(default_factory=list, description="Blocked tasks")
    
    # Results and metrics
    result: Optional[Dict[str, Any]] = None
    performance_metrics: Optional[Dict[str, Any]] = None
    error_summary: Optional[Dict[str, Any]] = None


class WorkflowTaskAssignmentRequest(BaseModel):
    """Request model for adding tasks to workflows."""
    task_id: str = Field(..., description="Task ID to add to workflow")
    dependencies: Optional[List[str]] = Field(None, description="Task dependencies")
    parallel_eligible: bool = Field(default=True, description="Can execute in parallel")
    critical_path: bool = Field(default=False, description="Part of critical path")


# ===============================================================================
# INTELLIGENT SCHEDULING MODELS
# ===============================================================================

class ScheduleRequest(BaseModel):
    """Request model for intelligent schedule generation."""
    model_config = ConfigDict(from_attributes=True)
    
    # Scheduling scope
    agent_id: Optional[str] = Field(None, description="Specific agent ID for scheduling")
    task_ids: Optional[List[str]] = Field(None, description="Specific task IDs to schedule")
    workflow_id: Optional[str] = Field(None, description="Workflow ID for scheduling")
    
    # Optimization parameters
    optimization_goal: str = Field(
        default="efficiency", 
        description="Optimization goal: efficiency, performance, resource_usage, availability"
    )
    time_horizon_hours: int = Field(
        default=24, ge=1, le=168, 
        description="Time horizon for schedule optimization"
    )
    
    # Constraints and preferences
    constraints: Dict[str, Any] = Field(
        default_factory=dict,
        description="Scheduling constraints and preferences"
    )
    resource_limits: Optional[Dict[str, float]] = Field(None, description="Resource utilization limits")
    blackout_periods: List[Dict[str, str]] = Field(
        default_factory=list, 
        description="Time periods to avoid"
    )
    
    # Advanced options
    enable_predictive_scheduling: bool = Field(default=True, description="Use predictive algorithms")
    learning_rate: float = Field(default=0.1, ge=0.01, le=1.0, description="ML learning rate")
    adaptation_window_hours: int = Field(default=6, ge=1, le=48, description="Adaptation window")
    
    @validator('optimization_goal')
    def validate_optimization_goal(cls, v):
        allowed_goals = ["efficiency", "performance", "resource_usage", "availability", "hybrid"]
        if v not in allowed_goals:
            raise ValueError(f"optimization_goal must be one of {allowed_goals}")
        return v


class ScheduleResponse(BaseModel):
    """Response model for intelligent scheduling operations."""
    model_config = ConfigDict(from_attributes=True)
    
    # Schedule identification
    schedule_id: str
    schedule_name: Optional[str] = None
    optimization_goal: str
    
    # Generated schedule
    schedule: Dict[str, Any] = Field(..., description="Generated schedule structure")
    task_assignments: List[Dict[str, Any]] = Field(default_factory=list, description="Task assignments")
    time_slots: List[Dict[str, Any]] = Field(default_factory=list, description="Time slot allocations")
    
    # Validation and performance
    validation_results: Dict[str, Any] = Field(default_factory=dict, description="Schedule validation")
    performance_predictions: Dict[str, Any] = Field(default_factory=dict, description="Performance predictions")
    confidence_score: float = Field(ge=0.0, le=1.0, description="Schedule confidence")
    
    # Timing information
    created_at: datetime
    valid_from: datetime
    expires_at: datetime
    
    # Metrics and insights
    efficiency_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    resource_utilization: Optional[Dict[str, float]] = None
    optimization_insights: List[Dict[str, Any]] = Field(default_factory=list)


class PatternAnalysisRequest(BaseModel):
    """Request model for activity pattern analysis."""
    agent_id: Optional[str] = Field(None, description="Agent ID for analysis")
    analysis_period_days: int = Field(
        default=7, ge=1, le=90,
        description="Number of days to analyze"
    )
    pattern_types: List[str] = Field(
        default_factory=lambda: ["activity", "sleep", "consolidation", "performance"],
        description="Types of patterns to analyze"
    )
    include_predictions: bool = Field(default=True, description="Include predictive insights")
    granularity: str = Field(default="hour", description="Analysis granularity: hour, day, week")


class ConflictResolutionRequest(BaseModel):
    """Request model for schedule conflict resolution."""
    conflict_resolution_strategy: str = Field(
        default="intelligent", 
        description="Strategy: intelligent, priority_based, optimal"
    )
    priority_weights: Dict[str, float] = Field(
        default_factory=lambda: {"efficiency": 0.4, "availability": 0.3, "performance": 0.3},
        description="Priority factor weights"
    )
    allow_rescheduling: bool = Field(default=True, description="Allow task rescheduling")
    force_resolution: bool = Field(default=False, description="Force conflict resolution")
    
    @validator('conflict_resolution_strategy')
    def validate_strategy(cls, v):
        allowed = ["intelligent", "priority_based", "first_come_first_serve", "optimal"]
        if v not in allowed:
            raise ValueError(f"strategy must be one of {allowed}")
        return v


# ===============================================================================
# TEAM COORDINATION MODELS
# ===============================================================================

class TeamCoordinationRequest(BaseModel):
    """Request model for team coordination operations."""
    coordination_type: str = Field(..., description="Type of coordination: assignment, balancing, communication")
    agents: List[str] = Field(..., min_items=1, description="Agent IDs for coordination")
    tasks: Optional[List[str]] = Field(None, description="Task IDs for coordination")
    
    # Coordination preferences
    load_balancing: bool = Field(default=True, description="Enable load balancing")
    skill_matching: bool = Field(default=True, description="Enable skill-based matching")
    real_time_updates: bool = Field(default=True, description="Enable real-time updates")
    
    # Communication settings
    notification_channels: List[str] = Field(
        default_factory=lambda: ["websocket", "redis"], 
        description="Notification channels"
    )
    priority_escalation: bool = Field(default=True, description="Enable priority escalation")


# ===============================================================================
# QUERY AND FILTER MODELS
# ===============================================================================

class TaskQueryParams(BaseModel):
    """Query parameters for task listing and filtering."""
    status: Optional[TaskStatus] = Field(None, description="Filter by task status")
    priority: Optional[TaskPriority] = Field(None, description="Filter by priority") 
    assigned_agent_id: Optional[str] = Field(None, description="Filter by assigned agent")
    task_type: Optional[TaskType] = Field(None, description="Filter by task type")
    
    # Date range filters
    created_after: Optional[datetime] = Field(None, description="Tasks created after date")
    created_before: Optional[datetime] = Field(None, description="Tasks created before date")
    
    # Pagination
    limit: int = Field(default=20, ge=1, le=100, description="Maximum results to return")
    offset: int = Field(default=0, ge=0, description="Number of results to skip")
    
    # Sorting
    sort_by: str = Field(default="created_at", description="Sort field")
    sort_order: str = Field(default="desc", description="Sort order: asc, desc")


class WorkflowQueryParams(BaseModel):
    """Query parameters for workflow listing and filtering."""
    status: Optional[WorkflowStatus] = Field(None, description="Filter by workflow status")
    priority: Optional[WorkflowPriority] = Field(None, description="Filter by priority")
    
    # Date range filters  
    created_after: Optional[datetime] = Field(None, description="Workflows created after date")
    created_before: Optional[datetime] = Field(None, description="Workflows created before date")
    
    # Pagination
    limit: int = Field(default=50, ge=1, le=100, description="Maximum results to return") 
    offset: int = Field(default=0, ge=0, description="Number of results to skip")


# ===============================================================================
# RESPONSE WRAPPERS AND COLLECTIONS
# ===============================================================================

class TaskListResponse(BaseModel):
    """Response model for task list operations."""
    tasks: List[TaskExecutionResponse]
    total: int = Field(ge=0, description="Total tasks matching criteria")
    offset: int = Field(ge=0, description="Results offset")
    limit: int = Field(ge=1, description="Results limit")
    
    # Summary statistics
    status_summary: Dict[str, int] = Field(default_factory=dict, description="Tasks by status")
    priority_summary: Dict[str, int] = Field(default_factory=dict, description="Tasks by priority")


class WorkflowListResponse(BaseModel):
    """Response model for workflow list operations."""
    workflows: List[WorkflowExecutionResponse]
    total: int = Field(ge=0, description="Total workflows matching criteria")
    offset: int = Field(ge=0, description="Results offset")
    limit: int = Field(ge=1, description="Results limit")


class OperationResponse(BaseModel):
    """Generic response model for operations."""
    success: bool = True
    message: str
    operation_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    details: Optional[Dict[str, Any]] = None
    
    
class HealthResponse(BaseModel):
    """Response model for health check operations."""
    service: str = "task_execution_api"
    healthy: bool = True
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Component health
    components: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    
    # Performance metrics
    metrics: Dict[str, Any] = Field(default_factory=dict)
    
    # Version and build info
    version: str = "2.0.0"
    build_info: Optional[Dict[str, Any]] = None