"""
Data Models for Multi-CLI Agent Coordination

This module defines all data models, message formats, and communication
structures used throughout the multi-CLI agent coordination system.

Key Components:
- AgentMessage: Standardized message format for inter-agent communication
- MessageType: Types of messages in the coordination protocol
- WorkflowDefinition: Structure for multi-agent workflows
- CoordinationRequest: Request format for agent coordination
- SystemStatus: Overall system health and status
"""

from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid

from .universal_agent_interface import (
    AgentType, TaskStatus, CapabilityType, HealthState,
    AgentTask, AgentResult, ExecutionContext, AgentCapability
)

# ================================================================================
# Communication and Messaging
# ================================================================================

class MessageType(str, Enum):
    """Types of messages in the agent coordination protocol"""
    # Task-related messages
    TASK_REQUEST = "task_request"
    TASK_RESPONSE = "task_response"
    TASK_UPDATE = "task_update"
    TASK_CANCEL = "task_cancel"
    
    # Coordination messages
    COORDINATION_REQUEST = "coordination_request"
    COORDINATION_RESPONSE = "coordination_response"
    HANDOFF_INITIATE = "handoff_initiate"
    HANDOFF_COMPLETE = "handoff_complete"
    
    # System messages
    AGENT_REGISTER = "agent_register"
    AGENT_UNREGISTER = "agent_unregister"
    HEALTH_CHECK = "health_check"
    HEALTH_RESPONSE = "health_response"
    CAPABILITY_QUERY = "capability_query"
    CAPABILITY_RESPONSE = "capability_response"
    
    # Workflow messages
    WORKFLOW_START = "workflow_start"
    WORKFLOW_STEP = "workflow_step"
    WORKFLOW_COMPLETE = "workflow_complete"
    WORKFLOW_ERROR = "workflow_error"
    
    # Monitoring messages
    STATUS_UPDATE = "status_update"
    PERFORMANCE_METRICS = "performance_metrics"
    ERROR_REPORT = "error_report"
    LOG_MESSAGE = "log_message"

class MessagePriority(str, Enum):
    """Message priority levels"""
    CRITICAL = "critical"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"

@dataclass
class MessageMetadata:
    """Metadata for agent messages"""
    sender_id: str
    sender_type: AgentType
    recipient_id: Optional[str] = None
    recipient_type: Optional[AgentType] = None
    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    request_id: Optional[str] = None
    reply_to: Optional[str] = None
    priority: MessagePriority = MessagePriority.NORMAL
    ttl_seconds: int = 300  # 5 minutes default
    routing_hints: Dict[str, str] = field(default_factory=dict)
    security_context: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AgentMessage:
    """Standardized message format for agent communication"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: MessageType = MessageType.TASK_REQUEST
    content: Dict[str, Any] = field(default_factory=dict)
    metadata: MessageMetadata = field(default_factory=MessageMetadata)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        """Validate message parameters"""
        if not self.content and self.type not in [MessageType.HEALTH_CHECK, MessageType.CAPABILITY_QUERY]:
            raise ValueError("Message content cannot be empty for this message type")

    def create_reply(self, message_type: MessageType, content: Dict[str, Any]) -> 'AgentMessage':
        """Create a reply message to this message"""
        reply_metadata = MessageMetadata(
            sender_id=self.metadata.recipient_id or "unknown",
            sender_type=self.metadata.recipient_type or AgentType.PYTHON_AGENT,
            recipient_id=self.metadata.sender_id,
            recipient_type=self.metadata.sender_type,
            correlation_id=self.metadata.correlation_id,
            request_id=self.id,
            reply_to=self.id,
            priority=self.metadata.priority
        )
        
        return AgentMessage(
            type=message_type,
            content=content,
            metadata=reply_metadata
        )

# ================================================================================
# Workflow and Coordination
# ================================================================================

class WorkflowStepType(str, Enum):
    """Types of workflow steps"""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    CONDITIONAL = "conditional"
    LOOP = "loop"
    FORK = "fork"
    JOIN = "join"

@dataclass
class WorkflowStep:
    """Individual step in a workflow"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    type: WorkflowStepType = WorkflowStepType.SEQUENTIAL
    agent_type: Optional[AgentType] = None
    agent_id: Optional[str] = None
    task: Optional[AgentTask] = None
    dependencies: List[str] = field(default_factory=list)  # Step IDs this step depends on
    conditions: Dict[str, Any] = field(default_factory=dict)
    timeout_seconds: int = 300
    retry_policy: Dict[str, Any] = field(default_factory=dict)
    error_handling: Dict[str, Any] = field(default_factory=dict)
    
    def can_execute(self, completed_steps: List[str]) -> bool:
        """Check if this step can execute based on dependencies"""
        return all(dep in completed_steps for dep in self.dependencies)

@dataclass
class WorkflowDefinition:
    """Definition of a multi-agent workflow"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    version: str = "1.0"
    steps: List[WorkflowStep] = field(default_factory=list)
    global_context: ExecutionContext = field(default_factory=ExecutionContext)
    global_timeout_seconds: int = 3600  # 1 hour
    error_policy: str = "fail_fast"  # fail_fast, continue, retry
    notification_settings: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self) -> List[str]:
        """Validate workflow definition and return any errors"""
        errors = []
        
        if not self.steps:
            errors.append("Workflow must have at least one step")
        
        step_ids = [step.id for step in self.steps]
        for step in self.steps:
            for dep in step.dependencies:
                if dep not in step_ids:
                    errors.append(f"Step {step.id} has invalid dependency: {dep}")
        
        return errors
    
    def get_executable_steps(self, completed_steps: List[str]) -> List[WorkflowStep]:
        """Get steps that can be executed given completed steps"""
        return [step for step in self.steps 
                if step.id not in completed_steps and step.can_execute(completed_steps)]

class WorkflowStatus(str, Enum):
    """Workflow execution status"""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class WorkflowExecution:
    """Runtime state of workflow execution"""
    workflow_id: str
    execution_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: WorkflowStatus = WorkflowStatus.PENDING
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    current_step: Optional[str] = None
    completed_steps: List[str] = field(default_factory=list)
    failed_steps: List[str] = field(default_factory=list)
    step_results: Dict[str, AgentResult] = field(default_factory=dict)
    error_message: Optional[str] = None
    execution_context: Optional[ExecutionContext] = None
    
    def mark_step_completed(self, step_id: str, result: AgentResult) -> None:
        """Mark a step as completed with its result"""
        if step_id not in self.completed_steps:
            self.completed_steps.append(step_id)
        self.step_results[step_id] = result
    
    def mark_step_failed(self, step_id: str, error: str) -> None:
        """Mark a step as failed with error message"""
        if step_id not in self.failed_steps:
            self.failed_steps.append(step_id)
        if step_id in self.completed_steps:
            self.completed_steps.remove(step_id)

# ================================================================================
# Coordination and Handoff
# ================================================================================

@dataclass
class CoordinationRequest:
    """Request for multi-agent coordination"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: str = "sequential"  # sequential, parallel, pipeline
    requesting_agent: str = ""
    target_agents: List[str] = field(default_factory=list)
    tasks: List[AgentTask] = field(default_factory=list)
    coordination_context: ExecutionContext = field(default_factory=ExecutionContext)
    priority: int = 5
    timeout_seconds: int = 1800  # 30 minutes
    requirements: Dict[str, Any] = field(default_factory=dict)
    
class CoordinationStatus(str, Enum):
    """Status of coordination request"""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class CoordinationResponse:
    """Response to coordination request"""
    request_id: str
    responding_agent: str
    status: CoordinationStatus
    estimated_completion: Optional[datetime] = None
    resource_requirements: Dict[str, Any] = field(default_factory=dict)
    conditions: List[str] = field(default_factory=list)
    alternative_proposal: Optional[CoordinationRequest] = None
    reason: Optional[str] = None

@dataclass
class HandoffPackage:
    """Package for context handoff between agents"""
    from_agent: str
    to_agent: str
    task_id: str
    context: ExecutionContext
    partial_results: Dict[str, Any] = field(default_factory=dict)
    work_products: List[str] = field(default_factory=list)  # File paths
    status_summary: str = ""
    next_steps: List[str] = field(default_factory=list)
    handoff_metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)

# ================================================================================
# System Status and Monitoring
# ================================================================================

@dataclass
class SystemStatus:
    """Overall system health and status"""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    overall_health: HealthState = HealthState.HEALTHY
    active_agents: int = 0
    total_agents: int = 0
    active_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    active_workflows: int = 0
    system_load: float = 0.0  # 0.0 to 1.0
    resource_usage: Dict[str, float] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    alerts: List[str] = field(default_factory=list)
    
    def calculate_health(self) -> HealthState:
        """Calculate overall system health based on metrics"""
        if self.system_load > 0.9 or self.failed_tasks > self.completed_tasks * 0.2:
            return HealthState.UNHEALTHY
        elif self.system_load > 0.7 or self.failed_tasks > self.completed_tasks * 0.1:
            return HealthState.DEGRADED
        else:
            return HealthState.HEALTHY

@dataclass
class AgentStatus:
    """Individual agent status for monitoring"""
    agent_id: str
    agent_type: AgentType
    health: HealthState
    current_tasks: List[str] = field(default_factory=list)
    completed_tasks: int = 0
    failed_tasks: int = 0
    last_activity: Optional[datetime] = None
    performance_score: float = 1.0  # 0.0 to 1.0
    capabilities: List[AgentCapability] = field(default_factory=list)
    load_score: float = 0.0  # 0.0 to 1.0
    
    def update_performance(self, success: bool, execution_time: float) -> None:
        """Update performance metrics based on task completion"""
        if success:
            self.completed_tasks += 1
        else:
            self.failed_tasks += 1
        
        # Simple performance score calculation
        total_tasks = self.completed_tasks + self.failed_tasks
        if total_tasks > 0:
            success_rate = self.completed_tasks / total_tasks
            # Weight recent performance more heavily
            self.performance_score = (self.performance_score * 0.8) + (success_rate * 0.2)

# ================================================================================
# Configuration and Settings
# ================================================================================

@dataclass
class AgentConfiguration:
    """Configuration for individual agents"""
    agent_id: str
    agent_type: AgentType
    cli_path: str = ""
    working_directory: str = ""
    environment_variables: Dict[str, str] = field(default_factory=dict)
    resource_limits: Dict[str, Any] = field(default_factory=dict)
    security_settings: Dict[str, Any] = field(default_factory=dict)
    performance_settings: Dict[str, Any] = field(default_factory=dict)
    logging_settings: Dict[str, Any] = field(default_factory=dict)
    custom_settings: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self) -> List[str]:
        """Validate configuration and return any errors"""
        errors = []
        
        if not self.agent_id:
            errors.append("Agent ID is required")
        
        if self.agent_type == AgentType.CLAUDE_CODE and not self.cli_path:
            errors.append("CLI path is required for Claude Code agents")
        
        return errors

@dataclass
class SystemConfiguration:
    """System-wide configuration"""
    redis_url: str = "redis://localhost:6379"
    database_url: str = "postgresql://localhost/beehive"
    log_level: str = "INFO"
    max_concurrent_tasks: int = 100
    default_task_timeout: int = 300
    default_workflow_timeout: int = 3600
    resource_monitoring_interval: int = 30
    health_check_interval: int = 60
    agent_configurations: Dict[str, AgentConfiguration] = field(default_factory=dict)
    security_policies: Dict[str, Any] = field(default_factory=dict)
    monitoring_settings: Dict[str, Any] = field(default_factory=dict)

# ================================================================================
# Error and Event Models
# ================================================================================

class ErrorSeverity(str, Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class ErrorReport:
    """Structured error reporting"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    severity: ErrorSeverity = ErrorSeverity.MEDIUM
    source_agent: str = ""
    source_component: str = ""
    error_type: str = ""
    error_message: str = ""
    stack_trace: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    resolution_steps: List[str] = field(default_factory=list)
    related_task_id: Optional[str] = None
    related_workflow_id: Optional[str] = None

class EventType(str, Enum):
    """System event types"""
    AGENT_REGISTERED = "agent_registered"
    AGENT_UNREGISTERED = "agent_unregistered"
    TASK_STARTED = "task_started"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    WORKFLOW_STARTED = "workflow_started"
    WORKFLOW_COMPLETED = "workflow_completed"
    COORDINATION_REQUESTED = "coordination_requested"
    HANDOFF_INITIATED = "handoff_initiated"
    ERROR_OCCURRED = "error_occurred"
    SYSTEM_HEALTH_CHANGE = "system_health_change"

@dataclass
class SystemEvent:
    """System event for monitoring and auditing"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: EventType = EventType.TASK_STARTED
    source: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    correlation_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None

# ================================================================================
# Utility Functions
# ================================================================================

def create_task_message(
    sender_id: str,
    sender_type: AgentType,
    task: AgentTask,
    recipient_id: Optional[str] = None
) -> AgentMessage:
    """Create a task request message"""
    metadata = MessageMetadata(
        sender_id=sender_id,
        sender_type=sender_type,
        recipient_id=recipient_id,
        priority=MessagePriority.HIGH if task.priority <= 3 else MessagePriority.NORMAL
    )
    
    return AgentMessage(
        type=MessageType.TASK_REQUEST,
        content={"task": task},
        metadata=metadata
    )

def create_result_message(
    sender_id: str,
    sender_type: AgentType,
    result: AgentResult,
    reply_to_message: AgentMessage
) -> AgentMessage:
    """Create a task result message"""
    return reply_to_message.create_reply(
        MessageType.TASK_RESPONSE,
        {"result": result}
    )

def create_coordination_message(
    sender_id: str,
    sender_type: AgentType,
    coordination_request: CoordinationRequest
) -> AgentMessage:
    """Create a coordination request message"""
    metadata = MessageMetadata(
        sender_id=sender_id,
        sender_type=sender_type,
        priority=MessagePriority.HIGH
    )
    
    return AgentMessage(
        type=MessageType.COORDINATION_REQUEST,
        content={"coordination_request": coordination_request},
        metadata=metadata
    )

# ================================================================================
# Constants
# ================================================================================

# Message size limits (bytes)
MAX_MESSAGE_SIZE = 10 * 1024 * 1024  # 10MB
MAX_CONTENT_SIZE = 8 * 1024 * 1024   # 8MB

# Timeout defaults (seconds)
DEFAULT_MESSAGE_TTL = 300  # 5 minutes
DEFAULT_COORDINATION_TIMEOUT = 1800  # 30 minutes
DEFAULT_HANDOFF_TIMEOUT = 600  # 10 minutes

# Performance thresholds
PERFORMANCE_THRESHOLDS = {
    "message_processing_ms": 100,
    "task_assignment_ms": 500,
    "coordination_response_ms": 2000,
    "system_health_check_ms": 1000
}