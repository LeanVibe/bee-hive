"""
Data Models for Enhanced Multi-CLI Orchestration

This module defines the core data structures for orchestrating multiple
CLI agents in complex workflows with intelligent routing and execution monitoring.
"""

import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Union
from dataclasses import dataclass, field
from pydantic import BaseModel, Field

from ..agents.universal_agent_interface import AgentType, AgentTask, AgentResult, AgentCapability

# ================================================================================
# Enums and Constants
# ================================================================================

class OrchestrationStatus(str, Enum):
    """Status of an orchestration request."""
    PENDING = "pending"
    PLANNING = "planning"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"

class WorkflowStepType(str, Enum):
    """Type of workflow step."""
    SEQUENTIAL = "sequential"      # Steps execute in order
    PARALLEL = "parallel"         # Steps execute concurrently
    CONDITIONAL = "conditional"   # Steps execute based on conditions
    LOOP = "loop"                # Steps repeat until condition met
    HUMAN_GATE = "human_gate"    # Requires human approval

class TaskPriority(str, Enum):
    """Task priority levels."""
    CRITICAL = "critical"         # Must execute immediately
    HIGH = "high"                # Execute as soon as possible
    NORMAL = "normal"            # Execute in normal queue order
    LOW = "low"                  # Execute when resources available
    BACKGROUND = "background"    # Execute during idle time

class RoutingStrategy(str, Enum):
    """Agent routing strategies."""
    BEST_FIT = "best_fit"                    # Route to agent with highest capability score
    LOAD_BALANCED = "load_balanced"          # Route to least loaded capable agent
    ROUND_ROBIN = "round_robin"              # Route in round-robin fashion
    STICKY_SESSION = "sticky_session"        # Keep related tasks on same agent
    COST_OPTIMIZED = "cost_optimized"        # Route to most cost-effective agent

# ================================================================================
# Core Orchestration Models
# ================================================================================

@dataclass
class OrchestrationRequest:
    """Request for orchestrating a complex task across multiple agents."""
    
    # Request identification
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    workflow_id: Optional[str] = None
    parent_request_id: Optional[str] = None
    
    # Task definition
    title: str = ""
    description: str = ""
    requirements: List[str] = field(default_factory=list)
    constraints: Dict[str, Any] = field(default_factory=dict)
    
    # Execution parameters
    priority: TaskPriority = TaskPriority.NORMAL
    routing_strategy: RoutingStrategy = RoutingStrategy.BEST_FIT
    max_execution_time_minutes: int = 60
    allow_parallel_execution: bool = True
    require_human_approval: bool = False
    
    # Resource limits
    max_agents: int = 5
    max_cost_units: float = 100.0
    preferred_agent_types: List[AgentType] = field(default_factory=list)
    excluded_agent_types: List[AgentType] = field(default_factory=list)
    
    # Context and data
    input_data: Dict[str, Any] = field(default_factory=dict)
    file_attachments: List[str] = field(default_factory=list)
    environment_variables: Dict[str, str] = field(default_factory=dict)
    
    # Tracking
    created_at: datetime = field(default_factory=datetime.utcnow)
    created_by: str = "system"
    tags: List[str] = field(default_factory=list)

@dataclass 
class TaskAssignment:
    """Assignment of a task to a specific agent."""
    
    # Assignment identification
    assignment_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    request_id: str = ""
    task_id: str = ""
    
    # Agent assignment
    agent_id: str = ""
    agent_type: AgentType = AgentType.CLAUDE_CODE
    assigned_at: datetime = field(default_factory=datetime.utcnow)
    
    # Execution details
    estimated_duration_minutes: int = 30
    estimated_cost_units: float = 10.0
    confidence_score: float = 0.8
    dependency_tasks: List[str] = field(default_factory=list)
    
    # Status tracking
    status: OrchestrationStatus = OrchestrationStatus.PENDING
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3

@dataclass
class OrchestrationResult:
    """Result of an orchestration request."""
    
    # Result identification
    request_id: str = ""
    result_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Execution summary
    status: OrchestrationStatus = OrchestrationStatus.PENDING
    success: bool = False
    error_message: Optional[str] = None
    
    # Task results
    task_assignments: List[TaskAssignment] = field(default_factory=list)
    completed_tasks: List[str] = field(default_factory=list)
    failed_tasks: List[str] = field(default_factory=list)
    
    # Output data
    output_data: Dict[str, Any] = field(default_factory=dict)
    files_created: List[str] = field(default_factory=list)
    files_modified: List[str] = field(default_factory=list)
    
    # Performance metrics
    total_execution_time_seconds: float = 0.0
    total_cost_units: float = 0.0
    agents_used: List[str] = field(default_factory=list)
    peak_concurrent_tasks: int = 0
    
    # Quality metrics
    success_rate: float = 0.0
    average_confidence: float = 0.0
    user_satisfaction_score: Optional[float] = None
    
    # Timing
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

# ================================================================================
# Workflow Models
# ================================================================================

@dataclass
class WorkflowStep:
    """Individual step in a workflow."""
    
    # Step identification
    step_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    
    # Step configuration
    step_type: WorkflowStepType = WorkflowStepType.SEQUENTIAL
    task_template: Dict[str, Any] = field(default_factory=dict)
    
    # Dependencies and conditions
    depends_on: List[str] = field(default_factory=list)
    condition: Optional[str] = None  # Python expression for conditional execution
    timeout_minutes: int = 30
    
    # Agent requirements
    required_agent_types: List[AgentType] = field(default_factory=list)
    required_capabilities: List[str] = field(default_factory=list)
    
    # Error handling
    continue_on_failure: bool = False
    retry_policy: Dict[str, Any] = field(default_factory=dict)
    
    # Human interaction
    requires_approval: bool = False
    approval_message: Optional[str] = None

@dataclass
class WorkflowDefinition:
    """Definition of a multi-step workflow."""
    
    # Workflow identification
    workflow_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    version: str = "1.0"
    
    # Workflow structure
    steps: List[WorkflowStep] = field(default_factory=list)
    global_timeout_minutes: int = 120
    
    # Execution configuration
    max_parallel_steps: int = 3
    failure_strategy: str = "stop_on_first_failure"  # or "continue_on_failure"
    routing_strategy: RoutingStrategy = RoutingStrategy.BEST_FIT
    
    # Resource limits
    max_total_cost_units: float = 500.0
    max_agents: int = 10
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    created_by: str = "system"
    tags: List[str] = field(default_factory=list)

@dataclass
class WorkflowExecution:
    """Runtime execution state of a workflow."""
    
    # Execution identification
    execution_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    workflow_id: str = ""
    request_id: str = ""
    
    # Current state
    status: OrchestrationStatus = OrchestrationStatus.PENDING
    current_step_id: Optional[str] = None
    completed_steps: List[str] = field(default_factory=list)
    failed_steps: List[str] = field(default_factory=list)
    
    # Step execution tracking
    step_executions: Dict[str, TaskAssignment] = field(default_factory=dict)
    step_results: Dict[str, Any] = field(default_factory=dict)
    
    # Timing and metrics
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    total_cost_units: float = 0.0

@dataclass
class WorkflowResult:
    """Result of workflow execution."""
    
    # Result identification
    execution_id: str = ""
    workflow_id: str = ""
    
    # Execution summary
    status: OrchestrationStatus = OrchestrationStatus.PENDING
    success: bool = False
    error_message: Optional[str] = None
    
    # Step results
    completed_steps: List[str] = field(default_factory=list)
    failed_steps: List[str] = field(default_factory=list)
    step_results: Dict[str, Any] = field(default_factory=dict)
    
    # Aggregated output
    final_output: Dict[str, Any] = field(default_factory=dict)
    all_files_created: List[str] = field(default_factory=list)
    all_files_modified: List[str] = field(default_factory=list)
    
    # Performance metrics
    total_execution_time_seconds: float = 0.0
    total_cost_units: float = 0.0
    steps_executed: int = 0
    agents_utilized: List[str] = field(default_factory=list)

# ================================================================================
# Agent Pool and Management Models
# ================================================================================

@dataclass
class AgentMetrics:
    """Performance metrics for an agent."""
    
    # Basic metrics
    total_tasks_completed: int = 0
    total_tasks_failed: int = 0
    average_execution_time_seconds: float = 0.0
    average_confidence_score: float = 0.0
    
    # Resource utilization
    current_cpu_usage: float = 0.0
    current_memory_usage_mb: float = 0.0
    active_tasks: int = 0
    queue_length: int = 0
    
    # Quality metrics
    success_rate: float = 0.0
    user_satisfaction_average: float = 0.0
    retry_rate: float = 0.0
    
    # Cost metrics
    total_cost_units_consumed: float = 0.0
    average_cost_per_task: float = 0.0
    cost_efficiency_score: float = 0.0
    
    # Timing
    last_activity: datetime = field(default_factory=datetime.utcnow)
    uptime_seconds: float = 0.0

@dataclass
class AgentPool:
    """Pool of available agents for orchestration."""
    
    # Pool identification
    pool_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "default"
    
    # Agent inventory
    available_agents: Dict[str, AgentType] = field(default_factory=dict)
    agent_capabilities: Dict[str, List[AgentCapability]] = field(default_factory=dict)
    agent_metrics: Dict[str, AgentMetrics] = field(default_factory=dict)
    
    # Pool configuration
    max_agents: int = 50
    load_balancing_strategy: RoutingStrategy = RoutingStrategy.LOAD_BALANCED
    health_check_interval_seconds: int = 60
    
    # Current state
    active_assignments: Dict[str, List[str]] = field(default_factory=dict)  # agent_id -> task_ids
    maintenance_mode_agents: Set[str] = field(default_factory=set)
    
    # Pool metrics
    total_pool_capacity: int = 0
    current_utilization: float = 0.0
    average_response_time_ms: float = 0.0

# ================================================================================
# Execution Status and Monitoring
# ================================================================================

@dataclass
class ExecutionStatus:
    """Real-time execution status for monitoring."""
    
    # Status identification
    request_id: str = ""
    execution_id: str = ""
    
    # Current state
    status: OrchestrationStatus = OrchestrationStatus.PENDING
    progress_percentage: float = 0.0
    current_phase: str = ""
    
    # Active tasks
    active_tasks: List[TaskAssignment] = field(default_factory=list)
    queued_tasks: List[str] = field(default_factory=list)
    
    # Progress tracking
    tasks_completed: int = 0
    tasks_failed: int = 0
    total_tasks: int = 0
    estimated_remaining_minutes: Optional[int] = None
    
    # Resource usage
    agents_in_use: List[str] = field(default_factory=list)
    cost_consumed: float = 0.0
    estimated_total_cost: float = 0.0
    
    # Quality indicators
    current_success_rate: float = 0.0
    average_confidence: float = 0.0
    
    # Timing
    started_at: Optional[datetime] = None
    last_updated: datetime = field(default_factory=datetime.utcnow)
    estimated_completion: Optional[datetime] = None

# ================================================================================
# Error and Recovery Models
# ================================================================================

@dataclass
class OrchestrationError:
    """Error information for orchestration failures."""
    
    # Error identification
    error_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    request_id: str = ""
    task_id: Optional[str] = None
    agent_id: Optional[str] = None
    
    # Error details
    error_type: str = ""
    error_message: str = ""
    error_code: Optional[str] = None
    stack_trace: Optional[str] = None
    
    # Context
    occurred_at: datetime = field(default_factory=datetime.utcnow)
    execution_phase: str = ""
    input_data: Dict[str, Any] = field(default_factory=dict)
    
    # Recovery information
    is_recoverable: bool = True
    suggested_recovery_action: Optional[str] = None
    retry_recommended: bool = True
    
    # Impact assessment
    affects_workflow: bool = False
    blocking_tasks: List[str] = field(default_factory=list)

@dataclass
class RecoveryAction:
    """Action taken to recover from an error."""
    
    # Action identification
    action_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    error_id: str = ""
    
    # Action details
    action_type: str = ""  # retry, reassign, skip, manual_intervention
    description: str = ""
    automated: bool = True
    
    # Execution
    executed_at: datetime = field(default_factory=datetime.utcnow)
    executed_by: str = "system"
    success: bool = False
    
    # Impact
    tasks_affected: List[str] = field(default_factory=list)
    cost_impact: float = 0.0